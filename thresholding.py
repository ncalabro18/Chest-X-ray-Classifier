"""
© 2026 Nicholas J. Calabro. All rights reserved.

Thresholds are applied to output scores to
create a binary decision on diseases (classes).

Dynamic thresholding is indended to sample the output of class model
as a probability. The model outputs 0.0 to 1.0 as a score of likelyhood
for each class. This is not a score nor is it linearly interpretable.
Therefore the previous web interface displaying a line graph with dual threhsolds
indicating a low or high likelyhood of the disease being present.

Thresholds can determine how accurate the model is at a specific point.
By bumping the number of thresholds until those points are close,
model output can be interpreted as a probability with some error.

"""
from dataclasses import dataclass

from sklearn.metrics import roc_curve
import torch
import numpy as np

from architecture import tta_predict
from classes import ALL_CLASSES, NUM_CLASSES, THRESHOLD_COUNT

THRESHOLD_MODIFIER_MIN = 1
THRESHOLD_MODIFIER_MAX = 2


# still per epoch level
class PerClassThreshold:
    def __init__(self):
        self.thresholds = np.zeros(NUM_CLASSES)
        self.metadata = []
        for i in range(NUM_CLASSES):
            self.metadata.append(ThresholdMetaData(
                sens=0.0,
                spec=0.0,
                ppv=0.0,
                npv=0.0,
                tp=0,
                tn=0,
                fn=0,
                fp=0,
                ece=0.0,
                alert_rate=0.0
            ))
            
        

@dataclass
class ThresholdMetaData:
    sens:       float # Sensitivity
    spec:       float # Specificity
    ppv:        float # Positive Predictive Values
    npv:        float # Negative Predictive Values
    tp:         int   # True Positives
    tn:         int   # True Negatives
    fn:         int   # False Negatives
    fp:         int   # False Positives
    ece:        float # Expected Calibration Error
    alert_rate: float
        

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

    thresholds = [PerClassThreshold() for _ in range(THRESHOLD_COUNT)]
    per_class_report = {cls: [] for cls in ALL_CLASSES}

    for c in range(NUM_CLASSES):
        col = (thresh_labels[:, c] >= 0.5).astype(int)
        scores = thresh_probs[:, c]

        fpr, tpr, roc_thresholds = roc_curve(col, scores, drop_intermediate=False)
        fpr = fpr[1:]
        tpr = tpr[1:]
        roc_thresholds = roc_thresholds[1:]

        # sens anchor: penalises false negatives twice as much
        sens_idx = np.argmax(THRESHOLD_MODIFIER_MAX * tpr - fpr)
        sens_thresh = float(np.clip(roc_thresholds[sens_idx], 0.01, 0.99))

        # spec anchor
        ppv = tpr / (tpr + fpr + 1e-12)
        ppv_idx = np.argmax(ppv)
        ppv_thresh = float(np.clip(roc_thresholds[ppv_idx], 0.01, 0.99))

        if np.isclose(sens_thresh, ppv_thresh):
            pad = 0.02
            sens_thresh = max(0.01, sens_thresh - pad)
            ppv_thresh = min(0.99, ppv_thresh + pad)


        # Sweep is anchored at the two named endpoints, not at whichever
        # threshold happens to be numerically smaller this time
        candidate_thresholds = np.linspace(sens_thresh, ppv_thresh, THRESHOLD_COUNT)

        for i, t in enumerate(candidate_thresholds):
            preds = (scores >= t).astype(int)

            tp = ((col == 1) & (preds == 1)).sum()
            tn = ((col == 0) & (preds == 0)).sum()
            fn = ((col == 1) & (preds == 0)).sum()
            fp = ((col == 0) & (preds == 1)).sum()

            tmeta = ThresholdMetaData(
                tp=tp, tn=tn, fn=fn, fp=fp,
                spec=tn / (tn + fp + 1e-12),
                sens=tp / (tp + fn + 1e-12),
                ppv=tp / (tp + fp + 1e-12),
                npv=tn / (tn + fn + 1e-12),
                alert_rate=(tp + fp) / len(col),
                ece=expected_calibration_error(scores, thresh_labels[:, c])
            )

            thresholds[i].thresholds[c] = float(t)
            thresholds[i].metadata[c] = tmeta

            per_class_report[ALL_CLASSES[c]].append({
                "threshold_id": i,
                "threshold_value": float(t),
                "sens": float(tmeta.sens),
                "spec": float(tmeta.spec),
                "ppv": float(tmeta.ppv),
                "npv": float(tmeta.npv),
                "alert_rate": float(tmeta.alert_rate),
                "ece": float(tmeta.ece),
                "tp": int(tmeta.tp),
                "fp": int(tmeta.fp),
                "tn": int(tmeta.tn),
                "fn": int(tmeta.fn),
            })

            print(
                f"  {ALL_CLASSES[c]:<20s} "
                f"thr={t:.3f}({tmeta.sens:.3f}s/{tmeta.spec:.3f}sp) "
                f"ppv={tmeta.ppv:.3f} alert={tmeta.alert_rate:.3f} ECE={tmeta.ece:.4f}"
            )

    return thresholds, thresh_probs, thresh_labels, per_class_report

def compute_threshold_metrics(thresh_report, threshold_id=0):
    macro_sens, macro_spec, macro_ppv, macro_npv, macro_alert = [], [], [], [], []

    for cls in ALL_CLASSES:
        if cls not in thresh_report:
            continue
        m = thresh_report[cls][threshold_id]
        macro_sens.append(m["sens"])
        macro_spec.append(m["spec"])
        macro_ppv.append(m["ppv"])
        macro_npv.append(m["npv"])
        macro_alert.append(m["alert_rate"])


    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return {
        "val_thresh_sens":        _mean(macro_sens),
        "val_thresh_spec":        _mean(macro_spec),
        "val_thresh_ppv":         _mean(macro_ppv),
        "val_thresh_npv":         _mean(macro_npv),
        "val_thresh_alert_rate":  _mean(macro_alert),
    }

def print_thresholding_parameters():
    print("  Thresholding Parameters:")
    print("  THRESHOLD_COUNT", THRESHOLD_COUNT)
    print("  THRESHOLD_MODIFIER_MIN", THRESHOLD_MODIFIER_MIN)
    print("  THRESHOLD_MODIFIER_MAX", THRESHOLD_MODIFIER_MAX)
