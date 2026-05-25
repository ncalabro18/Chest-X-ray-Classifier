"""
© 2026 Nicholas J. Calabro. All rights reserved.

Setup:
 - Modify *_PATH variables to resolve to the local file setup
 - pip install -r requirements.txt
 - Ensure GPU is recognized, adjust BATCH_SIZE, NUM_WORKERS_*
    to a batch which will fit on the GPU VRAM and less workers
    than logical threads on the CPU


Model Training Script

Current Model:
Custom SimMIM-Pretrained SwinV2 backbone
MLP head including an auxiliary MLP to condition on view position

A WeightedRandomSampler is used to oversample from rare classes
Asymmetric Loss is used to focus on hard examples and
prevent overfitting to common classes
An Unfreeze schedule is utilized to prevent overwriting pretrained
weights with the randomized head.
Thresholding is optimized to penalize false negatives twice as
much as positives, though f1 and youden's j is printed


Note one definite inaccuracy: Hernia has such few entries
that the value test will not yield a valid result.
It is excluded from the mean AUC calculation.

Future improvements could include:
- including other datasets
- restricting diseases types to pleural or other subtypes
- use Google's NIH labels, less noisy but maybe more unbalanced

"""
import datetime
import os
import glob
import time
import numpy as np

from checkpoint import CheckpointFile
from dataloader import init_thresh_dataloader, init_train_dataloader, init_value_dataloader, print_dataloader_parameters
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer
from torch.amp import autocast
from tqdm import tqdm

from classes import (
    ALL_CLASSES, NO_FINDING_COL, NUM_CLASSES, 
)

from dataset import (
    make_value_tf, make_train_tf,
    print_dataset_parameters,   
    init_split, 
    CXR8Dataset,
)

from architecture import (
    CONSISTENCY_LOSS_WEIGHT, HEAD_WARMUP_START_FACTOR, IMAGE_SIZE, MultiClassifier, SwinWithView, AsymmetricLoss,
    Scheduler, PerClassTemperatureScaler,
    fit_temperature, 
    print_architecture_parameters, tta_predict,
)

from thresholding import fit_thresholds
from util import (
    compute_model_metrics, compute_threshold_metrics, init_device, init_metadata,
    print_util_parameters,
    PerClassCSVWriter, PerEpochCSVWriter
)

# From Microsoft's Github on SwinV2
from swin_transformer_v2 import SwinTransformerV2

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


### Paths ( *_PATH must resolve to train) ###
    
# Metadata filepath
METADATA_CSV_PATH = "../chest_xray_dataset/CXR8/Data_Entry_2017_v2020.csv"

# Expects unziped subdirectories containing original png image filenames
IMAGE_ROOT_PATH = "../chest_xray_dataset/CXR8/images_preprocessed"

# Self Supervized Learning Checkpoint filepath
SSL_CKPT_PATH = "simmim_backbone_epoch100.pth"

# Output paths for training data and the best model checkpoint
MODEL_OUTPUT_FILE = "swin_cxr8_best.pth"
PER_CLASS_CSV_FILE = "per_class.csv"
PER_EPOCH_CSV_FILE = "per_epoch.csv"

# RESUME_FILE = "checkpoint_epoch15.pth"

RESUME_FILE = None

# Training Control
# Controls when training ends

NUM_EPOCHS = 48
PATIENCE = 8


### Tuning Parameters ###


# Save frequency in epochs
CHECKPOINT_INTERVAL = 5

# seconds to sleep after training
# set to 0 if not concerned about hardware overheating
HARDWARE_PITY = 0



# Logging and tuning purposes
def print_train_parameters():
    print("Nicholas J. Calabro's Chest X-ray Classifier Test")
    desc = input("Enter Model Test Description: ")
    print("Start time: ", datetime.datetime.now())
    print("### Test Description ###")
    print(desc)
    print("###                  ###")
    print("Training parameters:")
    print("  METADATA_CSV_PATH", METADATA_CSV_PATH)
    print("  IMAGE_ROOT", IMAGE_ROOT_PATH)
    print("  SSL_CKPT", SSL_CKPT_PATH)
    print("  MODEL_OUTPUT_FILE", MODEL_OUTPUT_FILE)
    print("  NUM_EPOCHS", NUM_EPOCHS)

    print("  PATIENCE", PATIENCE)

    print("  HARDWARE_PITY", HARDWARE_PITY)

    print("  CHECKPOINT_INTERVAL", CHECKPOINT_INTERVAL)

### Main Model Driver ###
def main():
    # For logging & tuning purposes
    print_train_parameters()
    print_architecture_parameters()
    print_dataset_parameters()
    print_dataloader_parameters()
    print_util_parameters()
    

    # GPU prep, prints info, and init
    device = init_device()

    # Load metadata
    meta_df = init_metadata(METADATA_CSV_PATH)
    
    # Checkpoint file class
    ckpt_file = CheckpointFile(best_path=MODEL_OUTPUT_FILE, device=device)

    # Image lookup
    all_png = glob.glob(
        os.path.join(IMAGE_ROOT_PATH, "**", "*.png"), recursive=True)
    path_lookup = {os.path.basename(p): p for p in all_png}

    # Labels
    mlb = MultiLabelBinarizer(classes=ALL_CLASSES)
    label_matrix = mlb.fit_transform(meta_df["labels"])
    mask = meta_df["Image Index"].isin(path_lookup)
    meta_df = meta_df[mask].reset_index(drop=True)
    label_matrix = label_matrix[mask.values]


    # Split
    # split by Patient so value tests hasn't been trained on the same patients
    # This decreases auc of about 0.01 to 0.02 but is a more accurate test
    # Aggregate labels to patient level for stratified splitting
    train_idx, value_idx, thresh_idx = init_split(
        meta_df,
        label_matrix=label_matrix
    )

    
    # Transforms
    value_tf = make_value_tf(IMAGE_SIZE)
    train_tf = make_train_tf(IMAGE_SIZE)
    
    train_ds = CXR8Dataset(
        meta_df, label_matrix, train_idx, train_tf, path_lookup,
        verify_label_alignment=True
    )
    value_ds = CXR8Dataset(
        meta_df, label_matrix, value_idx, value_tf, path_lookup
    )
    thresh_ds = CXR8Dataset(
        meta_df,label_matrix, thresh_idx, value_tf, path_lookup
    )


    train_loader  =  init_train_dataloader(train_ds, train_idx, label_matrix)
    value_loader  =  init_value_dataloader(value_ds)
    thresh_loader =  init_thresh_dataloader(thresh_ds)


    classifier = MultiClassifier(
        device=device,
        train_idx=train_idx,
        label_matrix=label_matrix,
        backbone_path=SSL_CKPT_PATH
    )

    scheduler = Scheduler(
        classifier,
        max_epochs=NUM_EPOCHS
    )

    # Init LR    
    for group in classifier.optimizer.param_groups:
        if group.get("layer_idx", -1) == -1:
            group["lr"] = group["base_lr"] * HEAD_WARMUP_START_FACTOR
    
    # Training Loop
    def run_epoch(loader, train=True, eval_model=None):
        active_model = eval_model if (
            not train and eval_model is not None
        ) else classifier.model
        active_model.train() if train else active_model.eval()
        # model.train() if train else model.eval()
        total_loss = 0.0
        n_samples = 0
        all_logits, all_labels = [], []

        with torch.set_grad_enabled(train):
            for imgs, lbls, views in tqdm(
                    loader,
                    desc="train" if train else "val ",
                    leave=False,
                    mininterval=10.0
                ):
                imgs  = imgs.to(device, non_blocking=True)
                lbls  = lbls.to(device, non_blocking=True)
                views = views.to(device, non_blocking=True)
                n_samples += imgs.size(0)


                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    if train:
                        logits = active_model(imgs, views)
                        probs = torch.sigmoid(logits)
                    else:
                        probs = tta_predict(active_model, imgs, views)
                        logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))

                    loss = classifier.criterion(logits, lbls)

                    if train:
                        nf_prob = probs[:, NO_FINDING_COL]
                        disease_mask = torch.ones(NUM_CLASSES, device=probs.device, dtype=torch.bool)
                        disease_mask[NO_FINDING_COL] = False
                        disease_probs = probs[:, disease_mask]
                        consistency_loss = (nf_prob.unsqueeze(1) * disease_probs).mean()
                        loss = loss + CONSISTENCY_LOSS_WEIGHT * consistency_loss

                # Backward pass outside autocast
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), max_norm=1.0)
                    classifier.optimizer.step()
                    classifier.optimizer.zero_grad()
                    classifier.ema_model.update_parameters(classifier.raw_model)

                total_loss += loss.item() * imgs.size(0)
                all_logits.append(probs.float().cpu().detach())
                all_labels.append(lbls.detach().cpu())

        avg_loss = total_loss / n_samples
        probs  = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()

        aucs, per_class_auc, f1s, per_class_f1 = compute_model_metrics(
            labels,
            probs,
            best_thresh
        )


        
        g = torch.sigmoid(
            classifier.raw_model.stage_gates
        ).detach().cpu().tolist()
        final_temp = classifier.raw_model.stage_temps[0].abs().item()
        
        
        print([f"  stage_gates: {x:.8f}" for x in g])
        print( f"  stage_temp  (final): {round(final_temp, 3)}")
        
        mean_f1 = np.mean(f1s) if f1s else 0.0
        return avg_loss, np.mean(aucs), mean_f1, per_class_auc, per_class_f1, probs, labels


    # CSV log files    
    epoch_logger = PerEpochCSVWriter(PER_EPOCH_CSV_FILE)
    class_logger = PerClassCSVWriter(PER_CLASS_CSV_FILE)

    best_val = 0.0
    best_thresh = np.zeros(NUM_CLASSES)
    thresh_report = {}

    no_improve = 0
    start_epoch = 1

    if RESUME_FILE:
        if not os.path.exists(RESUME_FILE):
            raise FileNotFoundError(f"RESUME_FILE '{RESUME_FILE}' not found. Set to None to start fresh.")
        print(f"Resuming from {RESUME_FILE} ...")
        ckpt = torch.load(RESUME_FILE, map_location=device, weights_only=False)

        classifier.model.load_state_dict(ckpt["model"])
        classifier.ema_model.module.load_state_dict(ckpt["ema_model"])
        classifier.optimizer.load_state_dict(ckpt["optimizer"])

        best_val    = ckpt["best_val"]
        no_improve  = ckpt["no_improve"]
        start_epoch = ckpt["epoch"] + 1

        # Restore thresholds from best model file if it exists
        if os.path.exists(MODEL_OUTPUT_FILE):
            best_thresh = classifier.load_thresholds(MODEL_OUTPUT_FILE)
            print(f"  Restored thresholds from {MODEL_OUTPUT_FILE}")

        scheduler.unfreeze_scheduler.restore_to_epoch(start_epoch)

        print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}, no_improve={no_improve}")


    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        scheduler.step(epoch)


        ### Train ###
        tr_loss, tr_auc, tr_f1, _, _, _, _ = run_epoch(
            train_loader,
            train=True
        )


        if scheduler.is_swa():
            scheduler.swa_model.update_parameters(classifier.raw_model)

        # Reset GPU cache and sleep to cooldown hardware
        torch.cuda.empty_cache()
        time.sleep(HARDWARE_PITY)

        ### Evaluate with EMA model ###
        val_loss, val_auc,val_f1, per_class, per_class_f1, \
            val_probs, val_labels = run_epoch(
                value_loader,
                train=False,
                eval_model=classifier.ema_model.module
        )

        print("  Per-class F1:")
        for cls, f1 in sorted(per_class_f1.items(), key=lambda x: x[1]):
            print(f"    {cls:<20s} {f1:.3f}")
        print(f"  val_f1={val_f1:.4f}")

        print("  Per-class AUCs:")
        for cls, auc in sorted(per_class.items(), key=lambda x: x[1]):
            print(f"    {cls:<20s} {auc:.3f}")


        thresh_metrics = compute_threshold_metrics(thresh_report)
        val_thresh_sens       = thresh_metrics["val_thresh_sens"]
        val_thresh_spec       = thresh_metrics["val_thresh_spec"]
        val_thresh_ppv        = thresh_metrics["val_thresh_ppv"]
        val_thresh_npv        = thresh_metrics["val_thresh_npv"]
        val_thresh_alert_rate = thresh_metrics["val_thresh_alert_rate"]

        # Write CSV row for each file
        epoch_logger.write_epoch(
            epoch,
            tr_loss, tr_auc, tr_f1,
            val_loss, val_auc, val_f1,
            val_thresh_sens, val_thresh_spec, val_thresh_ppv,
            val_thresh_npv, val_thresh_alert_rate,
            per_class,
            best_thresh,
        )
        class_logger.write_all(
            epoch,
            thresh_report,
            per_class_auc=per_class
        )

        # Periodic full checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0:
            ckpt_file.save_periodic(
                f"checkpoint_epoch{epoch:02d}.pth",
                epoch, classifier, best_val, no_improve
            )


        if val_auc > best_val:
            best_val = val_auc
            no_improve = 0

            best_thresh, _, _, thresh_report = fit_thresholds(
                classifier.ema_model.module,
                thresh_loader,
                device
            )

            temperature_scaler = classifier.fit_and_attach_temperature(
                value_loader, NUM_CLASSES
            )
            ckpt_file.save(
                classifier=classifier,
                thresholds=best_thresh,
                temperature_scaler=temperature_scaler
            )

            print("  -> saved new best model")
        else:
            no_improve += 1
            print(f"  (no improvement {no_improve}/{PATIENCE})")

        # if view scale is approaching 2.0, its dominating classes where it may hurt AUC
        print("view scale = ", torch.sigmoid(
            classifier.raw_model.view_scale).item() * 2.0)
        print(f"  overfit_gap={tr_auc - val_auc:.4f}")


        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_auc={tr_auc:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")


        classifier.raw_model.print_stage_gates()

        head_lr = next(
            g["lr"] for g in classifier.optimizer.param_groups if g.get(
                "layer_idx") == -1)
        layer0_lr = next(
            g["lr"] for g in classifier.optimizer.param_groups if g.get(
                "layer_idx") == 0)
        print(f"  head_lr={head_lr:.2e}  layer0_lr={layer0_lr:.2e}")

        # Guard: don't stop before all unfreeze events have had time to stabilize
        if no_improve >= PATIENCE and epoch > scheduler.min_stop_epoch():
            print(f"Early stopping triggered at epoch {epoch}")
            break

    epoch_logger.close()
    class_logger.close()
    print("Done. Best val AUC:", round(best_val, 4))


    if scheduler.is_swa():
        torch.optim.swa_utils.update_bn(
            train_loader,
            scheduler.swa_model,
            device=device
        )

        # fit temperature on the SWA model before saving
        temperature_scaler = classifier.fit_and_attach_temperature(value_loader, NUM_CLASSES)

        ckpt_file.save_final(
            scheduler.swa_model.module.state_dict(),
            best_thresh,
            classifier.temperature_scaler
        )
        print(f"Temperature saved: {temperature_scaler.temps.mean().item():.4f}")
    else:
        ckpt = classifier.load_best_checkpoint(MODEL_OUTPUT_FILE)
        print("view_scale:", classifier.view_scale())
        print(f"Temperature saved: {temperature_scaler.temps.mean().item():.4f}")

    print("End time: ", datetime.datetime.now())


if __name__ == "__main__":
    main()
