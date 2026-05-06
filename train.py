"""
© 2026 Nicholas J. Calabro. All rights reserved.

Model Training Script

Current Model:
Custom SimMIM-Pretrained SwinV2 backbone
MLP head including an auxiliary MLP to condition on view position

A WeightedRandomSampler is used to oversample from rare classes
Asymmetric Loss is used to focus on hard examples and
prevent overfitting to common classes


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
import csv
import numpy as np

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import (
    make_value_tf, make_train_tf, worker_init_fn,
    print_dataset_parameters,   
    init_split, 
    CXR8Dataset,
    ALL_CLASSES
)

from architecture import (
    SwinWithView, AsymmetricLoss, UnfreezeScheduler,
    init_group_cosine, init_param_groups, 
    print_architecture_parameters, tta_predict
)

from util import (
    init_device, init_metadata, init_ckpt,
)

# From Microsoft's Github on SwinV2
from swin_transformer_v2 import SwinTransformerV2

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


### Paths (must resolve to train) ###
    
# Metadata filepath
METADATA_CSV_PATH = "../chest_xray_dataset/CXR8/Data_Entry_2017_v2020.csv"

# Expects unziped subdirectories containing original png image filenames
IMAGE_ROOT = "../chest_xray_dataset/CXR8/images_preprocessed"

# Self Supervized Learning Checkpoint filepath
SSL_CKPT = "simmim_backbone_epoch100.pth"

# Output paths for training data and the best model checkpoint
MODEL_OUTPUT_FILE = "swin_cxr8_best.pth"
LOG_OUTPUT_FILE   = "training_log.csv"


### Tuning Parameters ###

# Training Control
NUM_EPOCHS = 35
PATIENCE = 8

IMAGE_SIZE = 224

SWIN_WINDOW_SIZE = 7

# Learning Rates
BASE_LR = 7e-5
# No pretraining for head
# Multiply the BASE_LR to compensate

LR_LAYER_DECAY = 0.8
WEIGHT_DECAY = 1e-2

# Asymmetric Loss
GAMMA_POS  = 0.0
GAMMA_NEG = 4.0
ASYMMETRIC_CLIP = 0.05

# Sample extra from low appearing catagories
SAMPLER_POWER = 0.17

# Warmup backbone; previously trained
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.3
WARMUP_END_FACTOR = 1.0

# Relative to each group's base_lr, not global eta_min
ETA_MIN_RATIO = 0.09

# high decay prevents noise from destabilizing training
# may underfit if too high
EMA_DECAY = 0.9995

# Save frequency in epochs
CHECKPOINT_INTERVAL = 5


# column 1: epoch to unfreeze at
# column 2: layer index to unfreeze
UNFREEZE_SCHEDULE = {
    2: 3,
    4: 2,
    6: 1,
    8: 0,
}
UNFREEZE_WARMUP_EPOCHS = 3

# Initial warmup factor for newly unfrozen layers,
# relative to their base_lr
UNFREEZE_WARMUP_FACTOR = 0.1
# Bump LR for unfrozen layers by the end of its warmup
UNFREEZE_BUMP_FACTOR = 1.6


# seconds to sleep after training
# set to 0 if not concerned about hardware overheating
HARDWARE_PITY = 45


### Data Loader Parameters ###
# Ran without locking when workers were 2 and 2
# CPU will bottleneck less with higher workers, may deadlock (immediately)
BATCH_SIZE_VAL   = 16
BATCH_SIZE_TRAIN = 16
PREFETECH_FACTOR = 2

LOADER_WORKERS_TRAIN = 10
LOADER_WORKERS_VALUE = 2
PERSISTENT_WORKERS   = True


# Minimum number of value needed to evaluate a catagory's AUC
# to avoid unreliable estimates and early stopping
MIN_VAL_POSITIVES = 50


### Calculated Constants ###
NO_FINDING_COL = ALL_CLASSES.index("No Finding")
NUM_CLASSES = len(ALL_CLASSES)



# Logging and tuning purposes
def print_train_parameters():
    print("Start time: ", datetime.datetime.now())
    print("Training parameters:")
    print("  METADATA_CSV_PATH", METADATA_CSV_PATH)
    print("  IMAGE_ROOT", IMAGE_ROOT)
    print("  SSL_CKPT", SSL_CKPT)
    print("  MODEL_OUTPUT_FILE", MODEL_OUTPUT_FILE)
    print("  SAMPLER_POWER", SAMPLER_POWER)
    print("  WARMUP_EPOCHS", WARMUP_EPOCHS)
    print("  WARMUP_START_FACTOR", WARMUP_START_FACTOR)
    print("  WARMUP_END_FACTOR", WARMUP_END_FACTOR)
    print("  NUM_EPOCHS", NUM_EPOCHS)
    print("  IMAGE_SIZE", IMAGE_SIZE)
    print("  SWIN_WINDOW_SIZE", SWIN_WINDOW_SIZE)
    print("  BASE_LR", BASE_LR)
    print("  LR_LAYER_DECAY", LR_LAYER_DECAY)
    print("  ETA_MIN_RATIO", ETA_MIN_RATIO)
    print("  PATIENCE", PATIENCE)
    print("  ASYMMETRIC_CLIP", ASYMMETRIC_CLIP)
    print("  GAMMA_NEG", GAMMA_NEG)
    print("  GAMMA_POS", GAMMA_POS)
    print("  UNFREEZE_WARMUP_EPOCHS", UNFREEZE_WARMUP_EPOCHS)
    print("  UNFREEZE_WARMUP_FACTOR",UNFREEZE_WARMUP_FACTOR)
    print("  UNFREEZE_BUMP_FACTOR", UNFREEZE_BUMP_FACTOR)
    print("  WEIGHT_DECAY", WEIGHT_DECAY)
    print("  EMA_DECAY", EMA_DECAY)
    print("  BATCH_SIZE_VAL", BATCH_SIZE_VAL)
    print("  BATCH_SIZE_TRAIN", BATCH_SIZE_TRAIN)
    print("  HARDWARE_PITY", HARDWARE_PITY)
    print("  UNFREEZE_SCHEDULE", UNFREEZE_SCHEDULE )
    print("  TRAIN_LOADER_WORKERS", LOADER_WORKERS_TRAIN )
    print("  VALUE_LOADER_WORKERS", LOADER_WORKERS_VALUE)
    print("  PREFETECH_FACTOR", PREFETECH_FACTOR)
    print("  PERSISTENT_WORKERS", PERSISTENT_WORKERS)
    print("  CHECKPOINT_INTERVAL", CHECKPOINT_INTERVAL)
    print("  MIN_VAL_POSITIVES",   MIN_VAL_POSITIVES)


### Main Model Driver ###
def main():
    # For logging & tuning purposes
    print_train_parameters()
    print_dataset_parameters()
    print_architecture_parameters()

    # GPU prep and init
    device = init_device()

    # Load metadata
    df = init_metadata(METADATA_CSV_PATH)
    

    # Image lookup
    all_png = glob.glob(os.path.join(IMAGE_ROOT, "**", "*.png"), recursive=True)
    path_lookup = {os.path.basename(p): p for p in all_png}

    # Labels
    mlb = MultiLabelBinarizer(classes=ALL_CLASSES)
    label_matrix = mlb.fit_transform(df["labels"])
    mask = df["Image Index"].isin(path_lookup)
    df = df[mask].reset_index(drop=True)
    label_matrix = label_matrix[mask.values]

    # Split
    # split by Patient so value tests hasn't been trained on the same patients
    # This causes an auc decrease of about 0.01 but is a more accurate test
    # Aggregate labels to patient level for stratified splitting
    train_idx, value_idx = init_split(df, label_matrix=label_matrix)

    # Transforms
    value_tf = make_value_tf(IMAGE_SIZE)
    train_tf = make_train_tf(IMAGE_SIZE)
    
    train_ds = CXR8Dataset(
            df, label_matrix, train_idx, train_tf, path_lookup)
    value_ds = CXR8Dataset(
            df, label_matrix, value_idx, value_tf, path_lookup)
    
    # After building train_ds, verify alignment:
    img, lbl, view = train_ds[0]
    expected_label = label_matrix[train_idx[0]]
    assert np.array_equal(lbl.numpy(), expected_label), "Label mismatch!"

    # Sampler
    class_counts = label_matrix[train_idx].sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-6) ** SAMPLER_POWER
    sample_weights = (label_matrix[train_idx] * class_weights).sum(axis=1)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        sampler=sampler,
        num_workers=LOADER_WORKERS_TRAIN,
        worker_init_fn=worker_init_fn,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETECH_FACTOR,
    )

    val_loader = DataLoader(
        value_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=LOADER_WORKERS_VALUE,
        worker_init_fn=worker_init_fn,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETECH_FACTOR,
    )

    # Verify Window is valid for given image size and patch size
    patch_grid = IMAGE_SIZE // 4
    assert patch_grid % SWIN_WINDOW_SIZE == 0, (
        f"patch grid {patch_grid} not divisible by window_size {SWIN_WINDOW_SIZE} — "
        f"valid sizes: {[SWIN_WINDOW_SIZE * 4 * i for i in range(1, 20) if (SWIN_WINDOW_SIZE * 4 * i) >= 192]}"
    )

    # Initialize Backbone
    base = SwinTransformerV2(
        img_size=IMAGE_SIZE,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=SWIN_WINDOW_SIZE,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        # Param refers to gradient checkpoint, not SSL checkpoint
        use_checkpoint=False,
    )

    # Initialize Head
    model = SwinWithView(backbone=base, num_classes=NUM_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    with torch.no_grad():
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        v = torch.zeros(1, dtype=torch.long).to(device)
        out = raw_model(x, v)
        print("Model output shape:", out.shape)  # expect (1, NUM_CLASSES)


    layer_to_idx = {
        layer: i
        for i, layer in enumerate(raw_model.backbone.layers)
    }

    # Model Training Checkpoint
    init_ckpt(model=raw_model, path=SSL_CKPT)

    ema_model = AveragedModel(raw_model, multi_avg_fn=get_ema_multi_avg_fn(decay=EMA_DECAY))
                
    param_group = init_param_groups(raw_model, base_lr=BASE_LR, decay=LR_LAYER_DECAY,
                                    schedule=UNFREEZE_SCHEDULE)


    optimizer = torch.optim.AdamW(param_group, weight_decay=WEIGHT_DECAY)    

   
    criterion = AsymmetricLoss(
        gamma_pos=GAMMA_POS,
        gamma_neg=GAMMA_NEG,
        clip=ASYMMETRIC_CLIP,
        label_smooth=0.05,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=WARMUP_START_FACTOR,
        end_factor=WARMUP_END_FACTOR,
        total_iters=WARMUP_EPOCHS
    )
   
    # Unfreeze SwinV2 stages to warmup backbone
    unfreeze_scheduler = UnfreezeScheduler(
        layer_to_idx=layer_to_idx,
        optimizer=optimizer,
        schedule=UNFREEZE_SCHEDULE,
        warmup_epochs=UNFREEZE_WARMUP_EPOCHS
    )


    # Training Loop
    def run_epoch(loader, train=True, eval_model=None):
        active_model = eval_model if (not train and eval_model is not None) else model
        active_model.train() if train else active_model.eval()
        # model.train() if train else model.eval()
        total_loss = 0.0
        n_samples = 0
        all_logits, all_labels = [], []

        with torch.set_grad_enabled(train):
            for imgs, lbls, views in tqdm(loader, desc="train" if train else "val ", leave=False, mininterval=10.0):
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

                    loss = criterion(logits, lbls)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    ema_model.update_parameters(raw_model)

                total_loss += loss.item() * imgs.size(0)
                all_logits.append(probs.float().cpu().detach())
                all_labels.append(lbls.detach().cpu())


        avg_loss = total_loss / n_samples
        probs  = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()

        per_class_auc = {}
        aucs = []
        for c in range(labels.shape[1]):
            col = labels[:, c]
            n_pos = col.sum()
            if n_pos >= MIN_VAL_POSITIVES and n_pos < len(col):
                auc = roc_auc_score(col, probs[:, c])
                per_class_auc[ALL_CLASSES[c]] = round(auc, 3)
                if c != NO_FINDING_COL:
                    aucs.append(auc)
            elif n_pos > 0:
                # Still log it, just don't include in mean
                per_class_auc[ALL_CLASSES[c]] = round(roc_auc_score(col, probs[:, c]), 3)
        
        gates = torch.sigmoid(raw_model.stage_gates).detach().cpu().numpy()
        final_temp = raw_model.stage_temps[0].abs().item()
        print(f"  stage_gates (early): {np.round(gates, 3)}")
        print(f"  stage_temp  (final): {round(final_temp, 3)}")
        
        return avg_loss, np.mean(aucs), per_class_auc, probs, labels

    # Epoch Loop
    # Initialize CSV log
    with open(LOG_OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "tr_loss", "tr_auc", "val_loss", "val_auc"] + ALL_CLASSES)

    best_val = 0.0
    no_improve = 0
    group_warmup_remaining = {}
    best_thresh = np.zeros(NUM_CLASSES)

    for epoch in range(1, NUM_EPOCHS + 1):

        tr_loss, tr_auc, _, _, _ = run_epoch(train_loader, train=True)
        torch.cuda.empty_cache()
        time.sleep(HARDWARE_PITY)

        # Evaluate with EMA model
        val_loss, val_auc, per_class, val_probs, val_labels = run_epoch(
            val_loader,
            train=False,
            eval_model=ema_model.module
        )

        print("  Per-class AUCs:")
        for cls, auc in sorted(per_class.items(), key=lambda x: x[1]):
            print(f"    {cls:<20s} {auc:.3f}")

        # Write CSV row
        with open(LOG_OUTPUT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_auc, val_loss, val_auc] +
                            [per_class.get(c, "") for c in ALL_CLASSES])

        # Periodic full checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "no_improve": no_improve,
            }, f"checkpoint_epoch{epoch:02d}.pth")

        if val_auc > best_val:
            best_val = val_auc
            no_improve = 0

            # recompute thresholds
            new_thresh = np.zeros(NUM_CLASSES)

            for c in range(NUM_CLASSES):
                if val_labels[:, c].sum() < MIN_VAL_POSITIVES:
                    new_thresh[c] = 0.5
                    continue

                fpr, tpr, roc_thresholds = roc_curve(val_labels[:, c], val_probs[:, c])
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                new_thresh[c] = float(np.clip(roc_thresholds[best_idx], 0.01, 0.99))


            best_thresh = new_thresh.copy()
            print("Updated thresholds:", np.round(best_thresh, 3))

            # save checkpoint with thresholds
            torch.save({
                "model": ema_model.module.state_dict(),
                "thresholds": best_thresh
            }, MODEL_OUTPUT_FILE)

            print("  -> saved new best model + thresholds")
        else:
            no_improve += 1
            print(f"  (no improvement {no_improve}/{PATIENCE})")

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_auc={tr_auc:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")



        unfreeze_scheduler.step(group_warmup_remaining)
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            for group in optimizer.param_groups:
                group["lr"] = init_group_cosine(
                    group, epoch, NUM_EPOCHS, ETA_MIN_RATIO, WARMUP_EPOCHS
                )


        for group in optimizer.param_groups:
            lidx = group.get("layer_idx", -1)
            if lidx in group_warmup_remaining:
                epochs_done = UNFREEZE_WARMUP_EPOCHS - group_warmup_remaining[lidx] + 1
                scale = UNFREEZE_WARMUP_FACTOR + (UNFREEZE_BUMP_FACTOR - UNFREEZE_WARMUP_FACTOR) * (epochs_done / UNFREEZE_WARMUP_EPOCHS)
                group["lr"] *= scale


        for k in list(group_warmup_remaining):
            group_warmup_remaining[k] -= 1
            if group_warmup_remaining[k] <= 0:
                del group_warmup_remaining[k]

        head_lr = next(g["lr"] for g in optimizer.param_groups if g.get("layer_idx") == -1)
        layer0_lr = next(g["lr"] for g in optimizer.param_groups if g.get("layer_idx") == 0)
        print(f"  head_lr={head_lr:.2e}  layer0_lr={layer0_lr:.2e}")

        # Guard: don't stop before all unfreeze events have had time to stabilize
        if no_improve >= PATIENCE and epoch > max(UNFREEZE_SCHEDULE.keys()) + WARMUP_EPOCHS:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Done. Best val AUC:", round(best_val, 4))
    
    # Inspect learned view conditioning
    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    ckpt = torch.load(MODEL_OUTPUT_FILE, map_location=device)
    m.load_state_dict(ckpt["model"])
    thresholds = ckpt["thresholds"]

    print("view_scale:", torch.sigmoid(m.view_scale).item() * 2.0)
    print("End time: ", datetime.datetime.now())



if __name__ == "__main__":
    main()