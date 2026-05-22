def analyze_errors(val_probs, val_labels, thresholds, output_path="error_analysis.csv"):
    rows = []
    for c, cls in enumerate(ALL_CLASSES):
        preds = (val_probs[:, c] > thresholds[c]).astype(int)
        tp = ((preds == 1) & (val_labels[:, c] == 1)).sum()
        fp = ((preds == 1) & (val_labels[:, c] == 0)).sum()
        fn = ((preds == 0) & (val_labels[:, c] == 1)).sum()
        tn = ((preds == 0) & (val_labels[:, c] == 0)).sum()
        n_pos = val_labels[:, c].sum()
        rows.append({
            "class": cls,
            "n_positives": int(n_pos),
            "sensitivity": round(tp / (tp + fn + 1e-8), 3),
            "specificity": round(tn / (tn + fp + 1e-8), 3),
            "ppv": round(tp / (tp + fp + 1e-8), 3),
            "fn_rate": round(fn / (tp + fn + 1e-8), 3),
            "threshold": round(thresholds[c], 3),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)