"""
© 2026 Nicholas J. Calabro. All rights reserved.

Thresholds are applied to output probabilities to
create a binary decision on diseases (classes).

"""
from sklearn.metrics import roc_curve
import torch
import numpy as np

from classes import ALL_CLASSES, MIN_VAL_POSITIVES, NUM_CLASSES



# brightness/contrast jitter
# result just under 80% auc
def tta_predict(model, imgs, views):
    preds = []

    with torch.no_grad():
        # Original
        logits_orig = model(imgs, views)
        preds.append(logits_orig)

        # Horizontal flip
        imgs_flip = torch.flip(imgs, dims=[3])
        logits_flip = model(imgs_flip, views)
        preds.append(logits_flip)

    # average LOGITS, then sigmoid
    logits = torch.mean(torch.stack(preds), dim=0)
    return torch.sigmoid(logits)


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
    per_class_report = {}

    for c in range(NUM_CLASSES):
        col = (thresh_labels[:, c] >= 0.5).astype(int)
        if col.sum() < MIN_VAL_POSITIVES:
            continue

        fpr, tpr, roc_thresholds = roc_curve(col, thresh_probs[:, c])
        j_idx = np.argmax(tpr - fpr)
        w_idx = np.argmax(2 * tpr - fpr)

        j_thresh = roc_thresholds[j_idx]
        w_thresh = roc_thresholds[w_idx]

        t = float(np.clip(w_thresh, 0.01, 0.99))
        preds = (thresh_probs[:, c] >= t).astype(int)

        tp = ((col == 1) & (preds == 1)).sum()
        fp = ((col == 0) & (preds == 1)).sum()
        tn = ((col == 0) & (preds == 0)).sum()
        fn = ((col == 1) & (preds == 0)).sum()

        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        ppv  = tp / (tp + fp + 1e-12)
        npv  = tn / (tn + fn + 1e-12)
        alert_rate = (tp + fp) / len(col)

        ece = expected_calibration_error(thresh_probs[:, c], thresh_labels[:, c])

        print(
            f"  {ALL_CLASSES[c]:<20s} "
            f"thr={t:.3f} J={j_thresh:.3f} sens-w={w_thresh:.3f} "
            f"sens={sens:.3f} spec={spec:.3f} ppv={ppv:.3f} npv={npv:.3f} "
            f"alert={alert_rate:.3f} ECE={ece:.4f}"
        )

        best_thresh[c] = t
        per_class_report[ALL_CLASSES[c]] = {
            "threshold": t,
            "sens": float(sens),
            "spec": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "alert_rate": float(alert_rate),
            "ece": float(ece),
        }

    return best_thresh, thresh_probs, thresh_labels, per_class_report