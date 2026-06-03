"""
© 2026 Nicholas J. Calabro. All rights reserved.


Extra Utility Functions
- expected_calibration_error: 
        calculates global temperature miscalibration
- init_metadata: Loads and processes the metadata CSV,
        extracting labels and view positions
- init_ckpt: Loads a checkpoint file,
        handling both training and SSL formats
- init_device: Sets up the PyTorch device,
        preferring GPU if available,
        and prints device info            

"""

import csv
import os

from sklearn.metrics import f1_score, roc_auc_score

import torch
import pandas as pd
import numpy as np

from classes import ALL_CLASSES, MIN_VAL_POSITIVES, NO_FINDING_COL, NUM_CLASSES

# Check point file labels that aren't required
EXPECTED_MISSING = {
    "relative_coords_table",
    "relative_position_index",
    "attn_mask"
}




# Extracts labels and view_ids, and filters to PA/AP images only
def init_metadata(path):
    df = pd.read_csv(path)
    df = df[["Image Index", "Finding Labels",
             "View Position", "Patient ID"]].copy()
    df["view_id"] = df["View Position"].map({"PA": 0, "AP": 1}).fillna(0).astype(int)
    df = df[df["View Position"].isin(["PA", "AP"])].reset_index(drop=True)
    df["labels"] = df["Finding Labels"].str.split("|")
    
    return df

# Loads the SSL backbone checkpoint
# more of a utility class because it validates keys
# could move to architecture.py and use a more specific function here
def init_backbone(model, path):
    raw = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected checkpoint format: {type(raw)}")

    # Unwrap outer dict if present
    state = raw.get("model", raw)

    sample_keys = list(state.keys())[:6]
    print("State dict sample keys:", sample_keys)

    # Discriminate: training checkpoints (SwinWithView) have backbone.* keys
    # SSL/SimMIM checkpoints have bare patch_embed.*, layers.* keys
    is_training_ckpt = any(k.startswith("backbone.") for k in state.keys())

    if is_training_ckpt:
        print("Detected training checkpoint, loading full SwinWithView state")
        model.load_state_dict(state)
        return raw.get("epoch", None), raw.get("best_val", None)

    # SSL checkpoint - strip encoder prefix if present (some SimMIM releases use it)
    if any(k.startswith("encoder.") for k in state):
        print("Stripping 'encoder.' prefix")
        state = {k[len("encoder."):]: v
                 for k, v in state.items()
                 if k.startswith("encoder.")}

    ckpt = {
        k.replace("rpe_mlp", "cpb_mlp"): v
        for k, v in state.items()
        if "relative_coords_table" not in k
        and "relative_position_index" not in k
        and "attn_mask" not in k
        and k not in ("head.weight", "head.bias")
    }

    missing, unexpected = model.backbone.load_state_dict(ckpt, strict=False)
    unexpected_missing = [k for k in missing if not any(tag in k for tag in EXPECTED_MISSING)]

    if unexpected_missing:
        print(f"WARNING: {len(unexpected_missing)} unexpected missing keys:")
        for k in unexpected_missing:
            print(f"  {k}")
    else:
        print(f"SSL checkpoint loaded OK - {len(missing)} expected missing, {len(unexpected)} unexpected")

    return None, None

# Use GPU if available
# Displays device info and clears cache to avoid fragmentation issues
def init_device():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}]", torch.cuda.get_device_name(i))
    print("Using device:", device)
    print("**Device: ", device)
    print(torch.cuda.get_device_capability())
    torch.backends.cudnn.benchmark = True

    return device

class PerEpochCSVWriter:
    def __init__(self, path: str, append: bool = False):
        mode = "a" if (append and os.path.exists(path)) else "w"
        write_header = mode == "w"
        self._f = open(path, mode, newline="")
        self._writer = csv.writer(self._f)
        if write_header:
            self._writer.writerow([
                "epoch",
                "tr_loss", "tr_auc", "tr_f1",
                "val_loss", "val_auc", "val_f1",
                "val_thresh_sens", "val_thresh_spec", "val_thresh_ppv",
                "val_thresh_npv", "val_thresh_alert_rate"
            ] + [f"{c}_auc" for c in ALL_CLASSES] + [
            f"{c}_thresh" for c in ALL_CLASSES
            ])

    def write_epoch(
        self,
        epoch,
        tr_loss, tr_auc, tr_f1,
        val_loss, val_auc, val_f1,
        val_thresh_sens, val_thresh_spec, val_thresh_ppv,
        val_thresh_npv, val_thresh_alert_rate,
        per_class_auc,
        best_thresh,
    ):
        row = [
            epoch,
            tr_loss, tr_auc, tr_f1,
            val_loss, val_auc, val_f1,
            val_thresh_sens, val_thresh_spec, val_thresh_ppv,
            val_thresh_npv, val_thresh_alert_rate
        ]
        row += [per_class_auc.get(c, "") for c in ALL_CLASSES]
        row += [float(
            best_thresh[i]
        ) if best_thresh[i] > 0 else 0.5 for i in range(NUM_CLASSES)]
        self._writer.writerow(row)
        self.f.flush()

    def close(self):
        if not self.f.closed:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class PerClassCSVWriter:
    def __init__(self, path: str, append: bool = False):
        mode = "a" if (append and os.path.exists(path)) else "w"
        write_header = mode == "w"
        self._f = open(path, mode, newline="")
        self._writer = csv.writer(self._f)
        if write_header:
            self._writer.writerow([
                "_writer",
                "class",
                "threshold",
                "auc",
                "sens",
                "spec",
                "ppv",
                "npv",
                "alert_rate",
                "ece",
                "tp",
                "fp",
                "tn",
                "fn",
            ])

    def write_class_row(self, epoch, class_name, metrics, auc=None):
        self._writer.writerow([
            epoch,
            class_name,
            metrics.get("threshold", ""),
            auc if auc is not None else metrics.get("auc", ""),
            metrics.get("sens", ""),
            metrics.get("spec", ""),
            metrics.get("ppv", ""),
            metrics.get("npv", ""),
            metrics.get("alert_rate", ""),
            metrics.get("ece", ""),
            metrics.get("tp", ""),
            metrics.get("fp", ""),
            metrics.get("tn", ""),
            metrics.get("fn", ""),
        ])
        self.f.flush()

    def write_all(self, epoch, per_class_report, per_class_auc=None):
        per_class_auc = per_class_auc or {}
        for cls, metrics in per_class_report.items():
            self.write_class_row(epoch, cls, metrics, per_class_auc.get(cls))

    def close(self):
        if not self.f.closed:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()



def compute_model_metrics(labels, probs, best_thresh):
    
    # Calculate per-class auc
    per_class_auc = {}
    aucs = []
    for c in range(labels.shape[1]):
        col = (labels[:, c] >= 0.5).astype(int)
        n_pos = col.sum()
        if len(np.unique(col)) < 2:
            continue
        auc = roc_auc_score(col, probs[:, c])
        per_class_auc[ALL_CLASSES[c]] = round(auc, 3)
        if n_pos >= MIN_VAL_POSITIVES and c != NO_FINDING_COL:
            aucs.append(auc)
    
    # Calculate per-class f1 score
    per_class_f1 = {}
    f1s = []
    for c in range(labels.shape[1]):
        col = (labels[:, c] >= 0.5).astype(int)
        if col.sum() < MIN_VAL_POSITIVES:
            continue
        
        # Use the current best_thresh if available, else 0.5
        # best_thresh is captured from the outer scope
        t = best_thresh[c] if best_thresh is not None and best_thresh[c] > 0 else 0.5
        preds = (probs[:, c] >= t).astype(int)
        
        f1 = f1_score(col, preds, zero_division=0)
        per_class_f1[ALL_CLASSES[c]] = round(f1, 3)
        if c != NO_FINDING_COL:
            f1s.append(f1)
    return aucs, per_class_auc, f1s, per_class_f1



def compute_threshold_metrics(thresh_report):
    macro_sens = []
    macro_spec = []
    macro_ppv = []
    macro_npv = []
    macro_alert = []
    for cls in ALL_CLASSES:
        if cls not in thresh_report:
            continue
        m = thresh_report[cls]
        macro_sens.append(m["sens"])
        macro_spec.append(m["spec"])
        macro_ppv.append(m["ppv"])
        macro_npv.append(m["npv"])
        macro_alert.append(m["alert_rate"])
    return {
        "val_thresh_sens": float(np.mean(macro_sens)) if macro_sens else 0.0,
        "val_thresh_spec": float(np.mean(macro_spec)) if macro_spec else 0.0,
        "val_thresh_ppv": float(np.mean(macro_ppv)) if macro_ppv else 0.0,
        "val_thresh_npv": float(np.mean(macro_npv)) if macro_npv else 0.0,
        "val_thresh_alert_rate": float(np.mean(macro_alert)) if macro_alert else 0.0,
    }



# In theory, these parameters should not need to be printed,
# they should only effect tuning if it had been misconfigureds
# Adding anyways to be thorough
def print_util_parameters():
    print("Utility Parameters:")
    print("  EXPECTED_MISSING", EXPECTED_MISSING)