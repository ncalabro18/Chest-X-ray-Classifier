"""
© 2026 Nicholas J. Calabro. All rights reserved.

Thresholds are applied to output probabilities to
create a binary decision on diseases (classes).

"""
from sklearn.metrics import roc_curve
import torch
import numpy as np

from classes import ALL_CLASSES, NUM_CLASSES


SENS_FLOOR = 0.40


# Classes may benifit benfit from per-class temperatures rather
# than a global one
# High ECE -> should be per_class
def expected_calibration_error(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece


def fit_thresholds(ema_model, thresh_loader, device):
    thresh_probs, thresh_labels = [], []
    ema_model.eval()

    with torch.no_grad():
        for imgs, lbls, views in thresh_loader:
            imgs, views = imgs.to(device), views.to(device)
            thresh_probs.append(tta_predict(ema_model, imgs, views).cpu())
            thresh_labels.append(lbls)

    thresh_probs = torch.cat(thresh_probs).numpy()
    thresh_labels = torch.cat(thresh_labels).numpy()

    best_thresh = np.zeros(NUM_CLASSES)
    spec_thresh = np.zeros(NUM_CLASSES)   # NEW
    per_class_report = {}

    for c in range(NUM_CLASSES):
        col = (thresh_labels[:, c] >= 0.5).astype(int)

        fpr, tpr, roc_thresholds = roc_curve(col, thresh_probs[:, c])

        # exclude the artificial boundary point
        fpr = fpr[1:]
        tpr = tpr[1:]
        roc_thresholds = roc_thresholds[1:]
        
        j_idx = np.argmax(tpr - fpr)
        w_idx = np.argmax(2 * tpr - fpr)

        j_thresh = roc_thresholds[j_idx]
        w_thresh = roc_thresholds[w_idx]

        valid_mask = tpr >= SENS_FLOOR
        if valid_mask.any():
            # among valid points, maximize specificity (minimize fpr)
            spec_idx = np.argmin(fpr[valid_mask])
            # map back to original index
            spec_idx = np.where(valid_mask)[0][spec_idx]
        else:
            spec_idx = j_idx  # fallback to Youden's if floor unachievable
        spec_t = float(np.clip(roc_thresholds[spec_idx], 0.01, 0.99))

        t = float(np.clip(w_thresh, 0.01, 0.99))
        preds = (thresh_probs[:, c] >= t).astype(int)

        tp = ((col == 1) & (preds == 1)).sum()
        fp = ((col == 0) & (preds == 1)).sum()
        tn = ((col == 0) & (preds == 0)).sum()
        fn = ((col == 1) & (preds == 0)).sum()

        spec_preds = (thresh_probs[:, c] >= spec_t).astype(int)
        spec_tp = ((col == 1) & (spec_preds == 1)).sum()
        spec_fp = ((col == 0) & (spec_preds == 1)).sum()
        spec_tn = ((col == 0) & (spec_preds == 0)).sum()
        spec_fn = ((col == 1) & (spec_preds == 0)).sum()

        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        ppv  = tp / (tp + fp + 1e-12)
        npv  = tn / (tn + fn + 1e-12)
        alert_rate = (tp + fp) / len(col)

        ece = expected_calibration_error(thresh_probs[:, c], thresh_labels[:, c])

        print(
            f"  {ALL_CLASSES[c]:<20s} "
            f"thr={t:.3f}({sens:.3f}s/{spec:.3f}sp) "
            f"spec_thr={spec_t:.3f}({spec_tp/(spec_tp+spec_fn+1e-12):.3f}s/"
            f"{spec_tn/(spec_tn+spec_fp+1e-12):.3f}sp) "
            f"ppv={ppv:.3f} alert={alert_rate:.3f} ECE={ece:.4f}"
        )
        best_thresh[c] = t
        spec_thresh[c] = spec_t   # NEW
        per_class_report[ALL_CLASSES[c]] = {
            "threshold":      t,
            "spec_threshold": spec_t,   # NEW
            "sens": float(sens),
            "spec": float(spec),
            "ppv":  float(ppv),
            "npv":  float(npv),
            "alert_rate": float(alert_rate),
            "ece":  float(ece),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "spec_thresh_sens": float(spec_tp / (spec_tp + spec_fn + 1e-12)),
            "spec_thresh_spec": float(spec_tn / (spec_tn + spec_fp + 1e-12)),
            "spec_thresh_ppv":  float(spec_tp / (spec_tp + spec_fp + 1e-12)),
            "spec_thresh_npv":  float(spec_tn / (spec_tn + spec_fn + 1e-12)),
            "spec_thresh_alert_rate": float((spec_tp + spec_fp) / len(col)),
        }

    return best_thresh, spec_thresh, thresh_probs, thresh_labels, per_class_report

def compute_threshold_metrics(thresh_report):
    macro_sens, macro_spec, macro_ppv, macro_npv, macro_alert = [], [], [], [], []
    spec_sens, spec_spec, spec_ppv, spec_npv, spec_alert = [], [], [], [], []  # NEW

    for cls in ALL_CLASSES:
        if cls not in thresh_report:
            continue
        m = thresh_report[cls]
        macro_sens.append(m["sens"])
        macro_spec.append(m["spec"])
        macro_ppv.append(m["ppv"])
        macro_npv.append(m["npv"])
        macro_alert.append(m["alert_rate"])
        # NEW
        spec_sens.append(m["spec_thresh_sens"])
        spec_spec.append(m["spec_thresh_spec"])
        spec_ppv.append(m["spec_thresh_ppv"])
        spec_npv.append(m["spec_thresh_npv"])
        spec_alert.append(m["spec_thresh_alert_rate"])

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return {
        "val_thresh_sens":        _mean(macro_sens),
        "val_thresh_spec":        _mean(macro_spec),
        "val_thresh_ppv":         _mean(macro_ppv),
        "val_thresh_npv":         _mean(macro_npv),
        "val_thresh_alert_rate":  _mean(macro_alert),
        "val_spec_thresh_sens":        _mean(spec_sens),    # NEW
        "val_spec_thresh_spec":        _mean(spec_spec),    # NEW
        "val_spec_thresh_ppv":         _mean(spec_ppv),     # NEW
        "val_spec_thresh_npv":         _mean(spec_npv),     # NEW
        "val_spec_thresh_alert_rate":  _mean(spec_alert),   # NEW
    }