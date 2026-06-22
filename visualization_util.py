"""
© 2026 Nicholas J. Calabro. All rights reserved.
""" 
import base64
import io

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def _get_class_index(class_name: str, class_names: list) -> int:
    """Convert class name to index."""
    try:
        return class_names.index(class_name)
    except ValueError:
        raise ValueError(f"Class '{class_name}' not found in class names: {class_names}")

def sort_predictions_by_ppv_npv(csv_path: str, predictions: dict, max_viz_classes: int = 5):
    """
    Sort positive and negative predictions by PPV and NPV respectively.
    
    Args:
        csv_path: Path to per_class.csv (per_epoch.csv should be in same directory)
        predictions: Dict of {class_name: {'probability': float, 'positive': bool}}
        max_viz_classes: Maximum number of classes to return for each group
    
    Returns:
        Tuple of (positive_classes, negative_classes) sorted by PPV/NPV
    """
    import os
    
    # Load CSVs
    per_class = pd.read_csv(csv_path)
    per_epoch = pd.read_csv(os.path.join(os.path.dirname(csv_path), 'per_epoch.csv'))
    
    # Get best epoch (highest val_auc)
    best_epoch = per_epoch.sort_values('val_auc', ascending=False)['epoch'].iloc[0]
    
    # Filter to best epoch and create lookup dicts
    per_class_best = per_class[per_class['epoch'] == best_epoch]
    class_ppv = per_class_best.groupby('class')['ppv'].first().to_dict()
    class_npv = per_class_best.groupby('class')['npv'].first().to_dict()
    
    # Sort positives by PPV descending
    positive_classes = sorted(
        [(cls_name, cls_name) for cls_name in predictions
         if predictions[cls_name]['positive']],
        key=lambda x: class_ppv.get(x[1], 0),
        reverse=True
    )[:max_viz_classes]
    
    # Sort negatives by NPV descending
    negative_classes = sorted(
        [(cls_name, cls_name) for cls_name in predictions
         if not predictions[cls_name]['positive']],
        key=lambda x: class_npv.get(x[1], 0),
        reverse=True
    )[:max_viz_classes]
    
    return positive_classes, negative_classes



def _to_data_url(pil_img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil_img = pil_img.convert("RGB")   # JPEG doesn't support alpha
    pil_img.save(buf, format=fmt, quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def _overlay_heatmap_on_image(
    image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Resize heatmap to match image, apply JET colormap, blend."""
    img = image.convert("RGB")
    img_np = np.array(img)

    hm = np.clip(heatmap, 0, 1)
    hm = (hm * 255).astype(np.uint8)
    hm = cv2.resize(hm, (img_np.shape[1], img_np.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_np, 1.0 - alpha, hm_color, alpha, 0)
    return Image.fromarray(blended)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap AveragedModel / DataParallel to reach the inner SwinWithView."""
    inner = model
    if hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "module"):
        inner = inner.module
    return inner


def _tokens_to_heatmap(attn_1d: np.ndarray) -> np.ndarray:
    """
    Convert an arbitrary-length 1-D attention vector to a 2-D float32 array
    normalised to [0, 1]. Works regardless of whether token count is a perfect
    square by padding to the next rectangle then cropping.
    """
    n = attn_1d.shape[0]
    side = int(np.sqrt(n))
    h, w = side, side
    if h * w != n:
        w = int(np.ceil(n / h))
        padded = np.zeros(h * w, dtype=np.float32)
        padded[:n] = attn_1d
        attn_1d = padded
    return attn_1d.reshape(h, w).astype(np.float32)


def _find_last_conv_layer(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

EARLY_STAGE_TOKENS = 49

def generate_attention_map(
    model, input_tensor, view_tensor, original_image,
    class_idx=None, skip_forward=False,
) -> str:
    inner = _unwrap(model)
    inner.eval()

    if not skip_forward:
        with torch.no_grad():
            inner(input_tensor, view_tensor)

    if not hasattr(inner, "last_attention"):
        raise RuntimeError("Attention map not available from this model.")

    attn = inner.last_attention.detach().float().cpu()  # (B, num_classes, N_total)

    if class_idx is not None:
        attn_1d = attn[0, class_idx, :]
    else:
        attn_1d = attn[0].mean(dim=0)

    n_early = len(inner.stage_projs)                      # 3
    n_early_tokens = n_early * EARLY_STAGE_TOKENS         # 147
    n_final_tokens = attn_1d.shape[0] - n_early_tokens    # 144 (12×12)
    final_side = int(n_final_tokens ** 0.5)               # 12

    # Split early and final token blocks
    early_tokens = attn_1d[:n_early_tokens].reshape(n_early, EARLY_STAGE_TOKENS)
    final_tokens = attn_1d[n_early_tokens:].reshape(final_side, final_side).numpy()

    # Weight early stages by gate activations
    gate_acts = (1.0 + torch.tanh(inner.stage_gates)).detach().cpu()  # (n_early,)
    early_weighted = (early_tokens * gate_acts.unsqueeze(1)).sum(dim=0)  # (49,)
    early_map = early_weighted.numpy().reshape(7, 7)

    # Upsample both to a common size and combine
    early_up = cv2.resize(early_map, (final_side, final_side), interpolation=cv2.INTER_LANCZOS4)
    combined = 0.4 * early_up + 0.6 * final_tokens        # weight final stage higher

    combined -= combined.min()
    combined /= (combined.max() + 1e-8)
    combined = combined.astype(np.float32)

    return _to_data_url(_overlay_heatmap_on_image(original_image, combined, alpha=0.35))

def generate_gradient_saliency(
    model, input_tensor, view_tensor, original_image, class_idx,
    n_smooth=8, noise_level=0.10,
) -> str:
    inner = _unwrap(model)
    inner.eval()

    base = input_tensor.clone().float()          # (1, 3, H, W)
    noise_std = noise_level * (base.max() - base.min()).item()

    # Stack n_smooth noisy copies into a single batch - one forward+backward pass
    noisy = base.repeat(n_smooth, 1, 1, 1)       # (n_smooth, 3, H, W)
    noisy = noisy + torch.randn_like(noisy) * noise_std
    noisy.requires_grad_(True)

    view_batch = view_tensor.repeat(n_smooth)    # (n_smooth,)

    logits = inner(noisy, view_batch)            # (n_smooth, num_classes)
    score  = logits[:, class_idx].sum()
    inner.zero_grad()
    score.backward()

    # Average gradient magnitude across batch
    grad = noisy.grad.data.abs()                 # (n_smooth, 3, H, W)
    saliency = grad.max(dim=1).values.mean(dim=0).cpu().numpy().astype(np.float32)
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-8)

    return _to_data_url(_overlay_heatmap_on_image(
        original_image,
        saliency,
        alpha=0.55
    ))


def plot_confusion_matrices(labels, probs, thresholds, class_names, save_path="confusion_matrices.png"):
    """
    labels: (N, num_classes) numpy array
    probs:  (N, num_classes) numpy array  
    thresholds: (num_classes,) numpy array
    """
    preds = (probs >= thresholds).astype(int)
    
    mcm = multilabel_confusion_matrix(labels, preds)  # (num_classes, 2, 2)

    n_classes = len(class_names)
    cols = 4
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, (cm, name) in enumerate(zip(mcm, class_names)):
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=axes[i], colorbar=False)
        axes[i].set_title(name, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")