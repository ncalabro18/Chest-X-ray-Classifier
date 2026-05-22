import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


def _to_data_url(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


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


def generate_grad_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    view_tensor: torch.Tensor,
    original_image: Image.Image,
    class_idx: int = None,
) -> str:
    """
    Returns a base64 PNG data URL of the Grad-CAM heatmap overlaid on
    `original_image` at full resolution.
    """
    inner = _unwrap(model)
    inner.eval()

    target_layer = _find_last_conv_layer(inner)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")

    activations: dict = {}
    gradients: dict = {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        with torch.enable_grad():
            logits = inner(input_tensor, view_tensor)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if class_idx is None:
                class_idx = int(torch.argmax(logits[0]).item())
            score = logits[0, class_idx]
            inner.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)

            acts  = activations["value"]
            grads = gradients["value"]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            overlaid = _overlay_heatmap_on_image(original_image, cam, alpha=0.45)
            return _to_data_url(overlaid)
    finally:
        h1.remove()
        h2.remove()


def generate_attention_map(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    view_tensor: torch.Tensor,
    original_image: Image.Image,
) -> str:
    """
    Returns a base64 PNG data URL of the cross-attention heatmap overlaid on
    `original_image` at full resolution.
    """
    inner = _unwrap(model)
    inner.eval()

    # --- path 1: explicit forward_with_attention ---
    if hasattr(inner, "forward_with_attention"):
        with torch.no_grad():
            out = inner.forward_with_attention(input_tensor, view_tensor)
        if isinstance(out, tuple) and len(out) >= 2:
            attn = out[1]
            if torch.is_tensor(attn):
                attn = attn.detach().float().cpu()
                if attn.ndim == 4:
                    attn = attn.mean(dim=1)[0]
                elif attn.ndim == 3:
                    attn = attn[0].mean(dim=0)
                attn = attn - attn.min()
                attn = attn / (attn.max() + 1e-8)
                hm = _tokens_to_heatmap(attn.numpy())
                return _to_data_url(_overlay_heatmap_on_image(original_image, hm))

    # --- path 2: last_attention cached by SwinWithView.forward ---
    with torch.no_grad():
        inner(input_tensor, view_tensor)

    if hasattr(inner, "last_attention"):
        attn = inner.last_attention  # (B, num_classes, N_total)
        if torch.is_tensor(attn):
            attn = attn.detach().float().cpu()
            if attn.ndim == 3:
                attn = attn[0].mean(dim=0)   # → (N_total,)
            elif attn.ndim == 2:
                attn = attn.mean(dim=0)
            elif attn.ndim == 4:
                attn = attn.mean(dim=1)[0].mean(dim=0)
            attn = attn - attn.min()
            attn = attn / (attn.max() + 1e-8)
            hm = _tokens_to_heatmap(attn.numpy())
            return _to_data_url(_overlay_heatmap_on_image(original_image, hm))

    raise RuntimeError("Attention map not available from this model.")