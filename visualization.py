import cv2
import matplotlib.pyplot as plt


import torch
import numpy as np
import argparse
import cv2
from PIL import Image

from architecture import SwinWithView, init_device
from swin_transformer_v2 import SwinTransformerV2
from train_save import IMAGE_SIZE, MODEL_OUTPUT_FILE
from dataset import ALL_CLASSES, make_value_tf, PerImageStandardize

SAVE_PATH = "visualization.png"

def print_visualization_parameters():
    print("Visualization parameters:")
    print(f"  SAVE_PATH: {SAVE_PATH}")


def get_class_attention(model, img_tensor, class_idx, device, view_id=0):
    """
    Replicates the multi-scale cross-attention from SwinWithView.forward()
    and returns the per-token attention weight for class_idx reshaped to a 2D map.

    Because tokens come from 4 scales concatenated, we return only the
    final-stage slice of the attention map (highest semantic resolution)
    which aligns with GradCAM's spatial grid for easy comparison.
    """
    model.eval()
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        x = model.backbone.patch_embed(x)
        if model.backbone.ape:
            x = x + model.backbone.absolute_pos_embed
        x = model.backbone.pos_drop(x)

        early_feats = []
        for i, layer in enumerate(model.backbone.layers):
            x = layer(x)
            if i < len(model.backbone.layers) - 1:
                early_feats.append(x)

        # Track token counts per stage so we can slice later
        token_counts = [f.shape[1] for f in early_feats]

        stage_tokens = []
        for feat, proj in zip(early_feats, model.stage_projs):
            B, N, D = feat.shape
            h = w = int(N ** 0.5)
            feat_2d = feat.reshape(B, h, w, D).permute(0, 3, 1, 2)
            stage_tokens.append(proj(feat_2d).flatten(2).transpose(1, 2))

        x_normed = model.backbone.norm(x)
        x_normed = model.class_norm(x_normed)
        N_final = x_normed.shape[1]

        all_tokens = torch.cat([*stage_tokens, x_normed], dim=1)    # (1, N_total, C)

        q = model.class_queries[class_idx]                           # (C,)
        attn = (all_tokens @ q) * model.attn_scale                  # (1, N_total)
        attn = torch.softmax(attn, dim=-1).squeeze(0)               # (N_total,)

        # Slice out only the final-stage attention weights for the 2D map
        final_stage_attn = attn[-N_final:]                           # (N_final,)

    H = W = int(N_final ** 0.5)
    attn_map = final_stage_attn.reshape(H, W).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-8)
    return attn_map

class GradCAM:
    """
    GradCAM for the SwinWithView model.
    Hooks the final backbone stage output (patch tokens before norm).
    """
    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self._feats = None
        self._grads = None

        # Hook the output of the last backbone stage
        target = model.backbone.layers[-1]
        self._fwd_hook = target.register_forward_hook(self._save_feats)
        self._bwd_hook = target.register_full_backward_hook(self._save_grads)

    def _save_feats(self, module, input, output):
        # output may be a tuple depending on SwinV2 implementation
        self._feats = output[0] if isinstance(output, tuple) else output

    def _save_grads(self, module, grad_input, grad_output):
        self._grads = grad_output[0] if isinstance(grad_output, tuple) else grad_output[0]

    def __call__(self, img_tensor, class_idx, view_id=0):
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(self.device).requires_grad_(False)
        v = torch.tensor([view_id], dtype=torch.long, device=self.device)

        logits = self.model(x, v)               # (1, num_classes)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # feats / grads: (1, N, C)
        grads = self._grads.detach()            # (1, N, C)
        feats = self._feats.detach()            # (1, N, C)

        weights = grads.mean(dim=-1, keepdim=True)  # (1, N, 1)  -- GAP over C
        cam = (weights * feats).sum(dim=-1)     # (1, N)
        cam = cam.squeeze(0)                    # (N,)
        cam = torch.relu(cam)

        H = W = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(H, W).cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(img_np, heatmap_hw, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    img_np    : (H, W) or (H, W, 3) float32 in [0,1] or uint8
    heatmap_hw: (h, w) float in [0,1]
    Returns   : (H, W, 3) uint8 BGR
    """
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    H, W = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap_hw, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8   = (heatmap_resized * 255).clip(0, 255).astype(np.uint8)
    colored         = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.addWeighted(img_np, 1 - alpha, colored, alpha, 0)


def visualize_class(model, img_tensor, img_np, class_idx, device,
                    gradcam: GradCAM, view_id=0, save_path=SAVE_PATH):
    """
    Side-by-side: original | GradCAM | attention map
    img_tensor : (3, H, W) torch tensor (normalised)
    img_np     : (H, W) or (H, W, 3) raw image for display
    """
    class_name = ALL_CLASSES[class_idx]

    # Attention map (no grad needed)
    attn_map = get_class_attention(model, img_tensor, class_idx, device)  # (h, w)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-8)

    # GradCAM
    cam = gradcam(img_tensor, class_idx, view_id=view_id)

    # Overlays
    attn_overlay = overlay_heatmap(img_np.copy(), attn_map,
                                   colormap=cv2.COLORMAP_INFERNO)
    cam_overlay  = overlay_heatmap(img_np.copy(), cam,
                                   colormap=cv2.COLORMAP_JET)

    # Convert BGR -> RGB for matplotlib
    attn_rgb = cv2.cvtColor(attn_overlay, cv2.COLOR_BGR2RGB)
    cam_rgb  = cv2.cvtColor(cam_overlay,  cv2.COLOR_BGR2RGB)

    if img_np.ndim == 2:
        orig_disp = img_np
        orig_cmap = "gray"
    else:
        orig_disp = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
        orig_cmap = None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Class: {class_name}", fontsize=14)

    axes[0].imshow(orig_disp, cmap=orig_cmap); axes[0].set_title("Original");     axes[0].axis("off")
    axes[1].imshow(cam_rgb);                   axes[1].set_title("GradCAM");       axes[1].axis("off")
    axes[2].imshow(attn_rgb);                  axes[2].set_title("Attention Map"); axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def load_model(ckpt_path, device):
    base = SwinTransformerV2(
        img_size=IMAGE_SIZE,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
    )
    model = SwinWithView(backbone=base, num_classes=len(ALL_CLASSES)).to(device)

    print(f"Loading checkpoint: '{ckpt_path}'")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    thresholds = ckpt.get("thresholds", np.full(len(ALL_CLASSES), 0.5))

    print(f"Thresholds: {np.round(thresholds, 3)}")
    return model, thresholds


def load_image(img_path):
    """
    Replicates the CXR8Dataset __getitem__ pipeline exactly:
      PIL RGB -> numpy -> albumentations -> /255.0 -> PerImageStandardize

    Returns:
      img_tensor : (3, H, W) float32 torch tensor, standardized
      img_np     : (H, W) uint8 grayscale, for overlay display
    """
    pil = Image.open(img_path).convert("RGB")
    img_np_rgb = np.array(pil)                          # (H, W, 3) uint8

    tf = make_value_tf(IMAGE_SIZE)
    img_tensor = tf(image=img_np_rgb)["image"]          # (3, H, W) uint8 torch
    img_tensor = img_tensor.float() / 255.0
    img_tensor = PerImageStandardize()(img_tensor)      # (3, H, W) float32

    # Grayscale for overlay (after resize so it matches the heatmap resolution)
    img_resized = cv2.resize(img_np_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_gray    = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)  # (H, W) uint8

    return img_tensor, img_gray


def run_visualization(model, device, img_tensor, img_np, class_indices, view_id, save_prefix):
    gradcam = GradCAM(model, device)
    try:
        for class_idx in class_indices:
            class_name = ALL_CLASSES[class_idx]
            save_path  = f"{save_prefix}_{class_name.replace(' ', '_')}.png"
            print(f"  [{class_idx}] {class_name} -> {save_path}")
            visualize_class(
                model=model,
                img_tensor=img_tensor,
                img_np=img_np,
                class_idx=class_idx,
                device=device,
                gradcam=gradcam,
                view_id=view_id,
                save_path=save_path,
            )
    finally:
        gradcam.remove()


def main():
    parser = argparse.ArgumentParser(description="GradCAM + attention visualization for chest X-ray")
    parser.add_argument("image",
        help="Path to input PNG/JPG")
    parser.add_argument("--classes", nargs="+", type=int, default=None,
        help="Class indices to visualize. Defaults to top-3 predicted.")
    parser.add_argument("--view", type=int, choices=[0, 1], default=0,
        help="View position: 0=PA, 1=AP (default: 0)")
    parser.add_argument("--ckpt", default=MODEL_OUTPUT_FILE,
        help=f"Checkpoint path (default: {MODEL_OUTPUT_FILE})")
    parser.add_argument("--save-prefix", default="visualization",
        help="Output filename prefix; class name appended per file")
    args = parser.parse_args()

    device       = init_device()
    model, thresholds = load_model(args.ckpt, device)
    model.eval()

    img_tensor, img_np = load_image(args.image)

    if args.classes is not None:
        class_indices = args.classes
    else:
        with torch.no_grad():
            view_id = torch.tensor([args.view], dtype=torch.long, device=device)
            logits  = model(img_tensor.unsqueeze(0).to(device), view_id)
            probs   = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        predicted = np.where(probs > thresholds)[0].tolist()

        if predicted:
            predicted.sort(key=lambda i: probs[i], reverse=True)
            class_indices = predicted[:3]
        else:
            # Nothing cleared the threshold — fall back to top-3 raw prob
            class_indices = np.argsort(probs)[::-1][:3].tolist()
            print("No class exceeded threshold; showing top-3 by raw probability.")

        print("Classes:      ", [ALL_CLASSES[i] for i in class_indices])
        print("Probabilities:", {ALL_CLASSES[i]: round(float(probs[i]), 3) for i in class_indices})
        print("Thresholds:   ", {ALL_CLASSES[i]: round(float(thresholds[i]), 3) for i in class_indices})

    run_visualization(model, device, img_tensor, img_np,
                      class_indices, args.view, args.save_prefix)
    print("Done.")


if __name__ == "__main__":
    main()

