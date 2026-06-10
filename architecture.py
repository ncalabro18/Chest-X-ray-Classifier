"""
© 2026 Nicholas J. Calabro. All rights reserved.

Model Architecture
- SwinWithView: Custom model class that extends a SwinV2 backbone,
        incorporating multi-scale feature fusion,
        class queries, and view conditioning
- AsymmetricLoss: Custom loss function for multi-label classification,
        with separate focusing parameters for positives and negative,
        and optional label smoothing
- UnfreezeScheduler: Manages gradual unfreezing of backbone layers
- init_group_cosine: Function to calculate cosine-decayed learning rates for param groups
"""
import math
import os

from torch.optim.swa_utils import AveragedModel as SWAModel
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.nn as nn
import torch
import cv2
import numpy as np


from classes import ALL_CLASSES, NUM_CLASSES
from dataloader import BATCH_SIZE_TRAIN
from swin_transformer_v2 import SwinTransformerV2
from util import init_backbone


### Architecture Parameters ###


IMAGE_SIZE = 384
SWIN_WINDOW_SIZE = 12

FEATURE_DROPOUT    = 0.35
CLASSIFIER_DROPOUT = 0.15


# Learning Rates
BASE_LR = 6e-5
# No pretraining for head
# Multiply the BASE_LR to compensate

LR_LAYER_DECAY = 0.8
WEIGHT_DECAY = 0.06


HEAD_LR_MULTIPLIER = 2.7


STAGE_GATES_MULTIPLIER = 1.0 

VIEW_POSITION_SCALE = 0.2


# high decay prevents noise from destabilizing training
# may underfit if too high
# EMA_DECAY = 0.9992
EMA_DECAY = 0.9985

### Asymmetric Loss ###
# pos should stay > 0.5
ASYMMETRIC_GAMMA_POS = 1.0
ASYMMETRIC_GAMMA_NEG = 3.5
ASYMMETRIC_CLIP      = 0.03

ASYMMETRIC_LABEL_SMOOTH = 0.05

# may increase auc if increased; went down when @ 0
CONSISTENCY_LOSS_WEIGHT = 0.02

### Scheduler Parameters ###

# column 1: epoch to unfreeze at
# column 2: layer index to unfreeze
# UNFREEZE_SCHEDULE = {
#     1: 3,
#     3: 2,
#     5: 1,
#     8: 0
# }
UNFREEZE_SCHEDULE = {
    2: 3,
    4: 2,
    6: 1,
    8: 0
}
UNFREEZE_WARMUP_EPOCHS = 2

# Initial warmup factor for newly unfrozen layers,
# relative to their base_lr
UNFREEZE_WARMUP_FACTOR = 0.23
# Bump LR for unfrozen layers by the end of its warmup
# Highly dependant on schedule timing
UNFREEZE_BUMP_FACTOR = 1.0

HEAD_WARMUP_EPOCHS      = 2
HEAD_WARMUP_START_FACTOR = 0.3

# SWA - starts after final unfreeze warmup completes (epoch 20 + 4 warmup)
SWA_START_EPOCH = 14
SWA_LR          = 8e-6   # flat LR during SWA, below cosine floor

# Relative to each group's base_lr, not global eta_min
ETA_MIN_RATIO = 0.05


# label smoooth is increased for these disease classes
NOISY_CLASSES = ['Infiltration', 'Nodule', 'Pleural_Thickening']
PER_CLASS_GAMMA_NEG = {
    'Infiltration':       1.5,   # noisy - soften neg suppression
    'Nodule':             1.0,   # noisy + hard - slight softening
    'Pleural_Thickening': 2.0,
    'Fibrosis':          3.0,
}
PER_CLASS_GAMMA_POS = {
    # "Infiltration": 0.75,
    # "Nodule": 0.9,
    # "Pleural_Thickening": 0.9,
}
PER_CLASS_CLIP = {
    'Infiltration':       0.015,  # noisy - less clipping to match softer gamma_neg
    'Nodule':             0.02,
    'Pleural_Thickening': 0.02,
}

### End Tune Parameters  ###


### Calculated Constants ###
# Unlikely that these need to change
BASE_BATCH_SIZE  = 16
# keep set BASE_LR independant of BATCH_SIZE
BASE_LR_ADJUSTED = BASE_LR * (BATCH_SIZE_TRAIN / BASE_BATCH_SIZE) ** 0.5


# Model Wrapper
class SwinWithView(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        C = backbone.norm.normalized_shape[0]
        backbone.head = nn.Identity()
        self.backbone = backbone

        # Stage projections: each stage doubles channels (96→192→384→768)
        # Project all to C so they can be stacked and averaged
        with torch.no_grad():
            _x = torch.zeros(1, 3, backbone.patch_embed.img_size[0],
                                    backbone.patch_embed.img_size[1])
            _x = backbone.patch_embed(_x)
            if backbone.ape:
                _x = _x + backbone.absolute_pos_embed
            _x = backbone.pos_drop(_x)
            stage_dims = []
            for layer in backbone.layers:
                _x = layer(_x)
                stage_dims.append(_x.shape[-1])  # actual channel dim per stage
        print("Detected stage dims:", stage_dims)
    
        self.stage_projs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(d, C // 4, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(C // 4, C, 1),
        ) for d in stage_dims[:-1]])

        self.stage_gates = nn.Parameter(
            torch.zeros(len(stage_dims)-1).uniform_(-0.1, 0.1)
        )

        self.class_queries = nn.Parameter(torch.randn(num_classes, C) * 0.02)
        self.attn_scale = C ** -0.5
        
        self.view_embed = torch.nn.Embedding(2, 32)
        self.view_mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, C * 2)
        )
        self.view_scale = torch.nn.Parameter(
            torch.tensor(VIEW_POSITION_SCALE)
        )

        # Init

        nn.init.normal_(self.view_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.view_mlp[-1].bias)
        nn.init.trunc_normal_(self.class_queries, std=0.02)

        for proj_seq in self.stage_projs:
            for module in proj_seq.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(
                        module.weight.view(module.weight.size(0), -1)
                    )
                    nn.init.zeros_(module.bias)

        self.stage_temps = nn.Parameter(torch.ones(1))
        self.class_norm = nn.LayerNorm(C)

        self.class_head = nn.Sequential(
            nn.LayerNorm(C),
            nn.Dropout(FEATURE_DROPOUT),
            nn.Linear(C, 256),
            nn.GELU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(256, 1),
        )



    def forward(self, x, view_id):
        x = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        # Collect spatial feature maps from early stages
        early_feats = []
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x)
            if i < len(self.backbone.layers) - 1:
                early_feats.append(x)           # (B, N_i, D_i) - keep spatial, don't GAP

        # Project each early stage to C via Conv2d and flatten back to tokens
        # stage_projs has len(layers)-1 entries, one per early stage
        stage_tokens = []
        # In forward, replace the stage_tokens construction:
        for i, (feat, proj) in enumerate(zip(early_feats, self.stage_projs)):
            B, N, D = feat.shape
            h = w = int(N ** 0.5)
            feat_2d = feat.reshape(B, h, w, D).permute(0, 3, 1, 2)
            # Pool down to same spatial size as final stage (7x7)
            feat_2d = torch.nn.functional.adaptive_avg_pool2d(feat_2d, output_size=7)
            projected = proj(feat_2d).flatten(2).transpose(1, 2)
            gate = 1.0 + torch.tanh(self.stage_gates[i])
            stage_tokens.append(projected * gate)

        x_normed = self.backbone.norm(x)

        all_tokens = torch.cat(
            stage_tokens +
            [x_normed * torch.nn.functional.softplus(self.stage_temps[0])],
            dim=1
        )
        
        all_tokens = self.class_norm(all_tokens)
        
        # Class query cross-attention over all scales simultaneously
        B = all_tokens.size(0)
        Q = self.class_queries.unsqueeze(0).expand(B, -1, -1)       # (B, num_classes, C)
        attn = torch.bmm(Q, all_tokens.transpose(1, 2)) * self.attn_scale  # (B, num_classes, N_total)
        attn = torch.softmax(attn, dim=-1)
        # attention dropout
        attn = torch.nn.functional.dropout(attn, p=0.1, training=self.training)
        self.last_attention = attn.detach() 
        class_feats = torch.bmm(attn, all_tokens)                    # (B, num_classes, C)

        # View conditioning
        v = self.view_mlp(self.view_embed(view_id))                  # (B, C*2)
        gamma, beta = v.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        scale = torch.sigmoid(self.view_scale) * 2.0
        class_feats = class_feats * (1 + scale * gamma.unsqueeze(1)) + beta.unsqueeze(1)

        logits = self.class_head(class_feats).squeeze(-1)            # (B, num_classes)
        return logits
    
    def print_stage_gates(self):
        raw = self.stage_gates.detach().cpu()
        act = (1.0 + torch.tanh(self.stage_gates)).detach().cpu()
        print("stage_gates raw:", [f"{x:.6f}" for x in raw.tolist()])
        print("stage_gates act:", [f"{x:.6f}" for x in act.tolist()])


# Loss
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, clip=0.05,
                 eps=1e-8,  label_smooth=0.05, weight=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.label_smooth = label_smooth
        self.weight = weight


    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Clip negative probabilities
        clip = self.clip
        if isinstance(clip, torch.Tensor):
            clip = clip.to(logits.device)
        if (clip if isinstance(clip, float) else clip.max().item()) > 0:
            probs_neg = (1 - probs - clip).clamp(min=0)
        else:
            probs_neg = 1 - probs

        # gamma_pos / gamma_neg may be scalar or (C,) tensor
        gp = self.gamma_pos
        gn = self.gamma_neg
        if isinstance(gp, torch.Tensor):
            gp = gp.to(logits.device)
        if isinstance(gn, torch.Tensor):
            gn = gn.to(logits.device)
        # Asymmetric focusing
        pos_focal = (1 - probs) ** gp
        neg_focal = probs ** gn

        if self.label_smooth is not None:
            # label_smooth can be scalar or (num_classes,) tensor
            smooth = self.label_smooth
            if isinstance(smooth, torch.Tensor):
                smooth = smooth.to(targets.device)
            targets = targets * (1 - smooth) + 0.5 * smooth


        # Loss
        loss_pos = targets * torch.log(probs.clamp(min=self.eps)) * pos_focal
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps)) * neg_focal

        loss = -(loss_pos + loss_neg)          # (B, C)
        if self.weight is not None:
            loss = loss * self.weight
        return loss.mean()



class MultiClassifier:
    def __init__(self, backbone_path, device, label_matrix, train_idx):
        self.device = device
        self.temperature_scaler = None

        self.base = self._build_backbone()
        self.model = SwinWithView(
            backbone=self.base,
            num_classes=NUM_CLASSES
        ).to(self.device)
        self.raw_model = self._wrap_model(self.model)

        self._sanity_check_forward()
        init_backbone(self.model, backbone_path)

        self.ema_model = AveragedModel(
            model=self.raw_model,
            multi_avg_fn=get_ema_multi_avg_fn(decay=EMA_DECAY),
        )        

        self.param_group = init_param_groups(
            model=self.raw_model,
            base_lr=BASE_LR_ADJUSTED,
            decay=LR_LAYER_DECAY,
        )
        self.optimizer = torch.optim.AdamW(
            self.param_group,
            weight_decay=WEIGHT_DECAY,
        )

        self.criterion = self._build_criterion(label_matrix, train_idx)

    def _build_backbone(self):
        patch_grid = IMAGE_SIZE // 4
        assert patch_grid % SWIN_WINDOW_SIZE == 0, (
            f"patch grid {patch_grid} not divisible by window_size {SWIN_WINDOW_SIZE} - "
            f"valid sizes: {[SWIN_WINDOW_SIZE * 4 * i for i in range(1, 20) if (SWIN_WINDOW_SIZE * 4 * i) >= 192]}"
        )

        return SwinTransformerV2(
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
            use_checkpoint=False,
        )

   
    def _wrap_model(self, model):
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(model)  # update self.model
            return model                                # raw_model = unwrapped
        return model

    def _unwrapped_model(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _sanity_check_forward(self):
        with torch.no_grad():
            x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
            v = torch.zeros(1, dtype=torch.long, device=self.device)
            out = self.raw_model(x, v)
            print("Model output shape:", out.shape)

    def _build_criterion(self, label_matrix, train_idx):
        class_freq = label_matrix[train_idx].mean(axis=0)
        class_loss_weights = torch.tensor(
            1.0 / (class_freq + 1e-4) ** 0.3,
            dtype=torch.float32,
            device=self.device,
        )

        # Per-class label smooth, gamma_pos, and gamma_neg
        label_smooth = torch.full((NUM_CLASSES,), ASYMMETRIC_LABEL_SMOOTH)
        for cls in NOISY_CLASSES:
            if cls in ALL_CLASSES:
                label_smooth[ALL_CLASSES.index(cls)] = 0.10

        gamma_neg = torch.full((NUM_CLASSES,), float(ASYMMETRIC_GAMMA_NEG))
        gamma_pos = torch.full((NUM_CLASSES,), float(ASYMMETRIC_GAMMA_POS))
        for cls, val in PER_CLASS_GAMMA_NEG.items():
            if cls in ALL_CLASSES:
                gamma_neg[ALL_CLASSES.index(cls)] = val
        for cls, val in PER_CLASS_GAMMA_POS.items():
            if cls in ALL_CLASSES:
                gamma_pos[ALL_CLASSES.index(cls)] = val
        
        # Per-class AS clip
        clip = torch.full((NUM_CLASSES,), float(ASYMMETRIC_CLIP))
        for cls, val in PER_CLASS_CLIP.items():
            if cls in ALL_CLASSES:
                clip[ALL_CLASSES.index(cls)] = val

        return AsymmetricLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
            label_smooth=label_smooth,
            weight=class_loss_weights,
        )
            

    def load_best_checkpoint(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._unwrapped_model().load_state_dict(ckpt["model"])
        self.ema_model.module.load_state_dict(ckpt["model"])


        if ckpt.get("temperature") is not None:
            temps = torch.tensor(ckpt["temperature"], device=self.device)
            scaler = PerClassTemperatureScaler(len(temps)).to(self.device)
            scaler.temps = torch.nn.Parameter(temps)
            self.temperature_scaler = scaler

        return ckpt

    def load_thresholds(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw = ckpt.get("thresholds")
        if raw is None:
            print("WARNING: no thresholds in checkpoint; defaulting to 0.5")
            return np.full(NUM_CLASSES, 0.5, dtype=np.float32)
        return np.array(raw, dtype=np.float32)
    
    def load_spec_thresholds(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw = ckpt.get("spec_thresholds")
        if raw is None:
            print("WARNING: no spec_thresholds in checkpoint; defaulting to 0.5")
            return np.full(NUM_CLASSES, 0.5, dtype=np.float32)
        return np.array(raw, dtype=np.float32)

    def view_scale(self) -> float:
        return torch.sigmoid(self.ema_model.module.view_scale).item() * 2.0

    def fit_and_attach_temperature(self, val_loader, num_classes: int):
        scaler = PerClassTemperatureScaler(num_classes).to(self.device)
        fit_temperature(self._unwrapped_model(), val_loader, self.device, scaler)
        self.temperature_scaler = scaler
        return scaler
    
    def print_LRs(self):
        head_lr = next(
            g["lr"] for g in self.optimizer.param_groups if g.get(
                "layer_idx") == -1)
        layer0_lr = next(
            g["lr"] for g in self.optimizer.param_groups if g.get(
                "layer_idx") == 0)
        print(f"  head_lr={head_lr:.2e}  layer0_lr={layer0_lr:.2e}")

class PerClassTemperatureScaler(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.temps = nn.Parameter(torch.ones(num_classes))

    def forward(self, logits):
        return logits / self.temps.clamp(min=0.1)
    
class Scheduler:
    def __init__(self, classifier, max_epochs):
        self.max_epochs = max_epochs
        self.optimizer  = classifier.optimizer
        self.in_swa     = False
        self.raw_model = classifier.raw_model
        self.swa_model = SWAModel(self.raw_model)

        self.layer_to_idx = {
            layer: i
            for i, layer in enumerate(classifier.raw_model.backbone.layers)
        }
        # REMOVED: self.warmup_scheduler = LinearLR(...)
        self.unfreeze_scheduler = UnfreezeScheduler(
            layer_to_idx=self.layer_to_idx,
            optimizer=self.optimizer,
        )


    def step(self, epoch):
        self.unfreeze_scheduler.step(epoch)

        if epoch >= SWA_START_EPOCH:
            if not self.in_swa:
                print(f"  Switching to SWA flat LR={SWA_LR:.2e} at epoch {epoch}")
                for group in self.optimizer.param_groups:
                    group["lr"] = SWA_LR
                self.in_swa = True

        elif epoch <= HEAD_WARMUP_EPOCHS:
            # Warm up head only - backbone is frozen so its LR doesn't matter yet
            factor = HEAD_WARMUP_START_FACTOR + (
                1.0 - HEAD_WARMUP_START_FACTOR
            ) * (epoch / HEAD_WARMUP_EPOCHS)
            for group in self.optimizer.param_groups:
                if group.get("layer_idx", -1) == -1:
                    group["lr"] = group["base_lr"] * factor

        else:
            for group in self.optimizer.param_groups:
                group["lr"] = init_group_cosine(
                    group, epoch, self.max_epochs, ETA_MIN_RATIO, warmup_epochs=0
                )
        
        
        self.unfreeze_scheduler.apply_scales()


    def min_stop_epoch(self):
        return max(SWA_START_EPOCH, max(UNFREEZE_SCHEDULE))

    def is_swa(self):
        return self.in_swa


class UnfreezeScheduler:
    def __init__(self, layer_to_idx, optimizer):
        self.optimizer = optimizer
        self.layer_to_idx = layer_to_idx
        self.group_warmup_remaining = {}
        self.schedule = UNFREEZE_SCHEDULE
        self.warmup_epochs = UNFREEZE_WARMUP_EPOCHS
        # freeze layers
        for layer, idx in layer_to_idx.items():
            for p in layer.parameters():
                p.requires_grad = False

    # Unfreeze layers per schedule
    def step(self, epoch):
        newly_unfrozen = set()
        if epoch in self.schedule:
            if self.group_warmup_remaining:
                print(f"WARNING: Unfreezing at epoch {epoch}, "
                      f"but warmup still active for layers: "
                      f"{list(self.group_warmup_remaining.keys())}")
            
            threshold = self.schedule[epoch]
            # Check no warmup is still in progress
            for layer, idx in self.layer_to_idx.items():
                if idx >= threshold:
                    for p in layer.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            newly_unfrozen.add(idx)
            for group in self.optimizer.param_groups:
                lidx = group.get("layer_idx", -1)
                if lidx in newly_unfrozen:
                    # Use cosine-decayed peer LR, not the original base_lr
                    ref = next(g for g in self.optimizer.param_groups
                            if g.get("layer_idx") == -1)
                    cosine_scale = ref["lr"] / ref["base_lr"]
                    group["lr"] = group["base_lr"] * cosine_scale
                    self.group_warmup_remaining[lidx] = self.warmup_epochs


    def apply_scales(self):
        # Match the cosine position of the always-live head group
        ref = next(g for g in self.optimizer.param_groups
                if g.get("layer_idx") == -1)
        cosine_scale = ref["lr"] / ref["base_lr"]

        for group in self.optimizer.param_groups:
            lidx = group.get("layer_idx", -1)
            if lidx in self.group_warmup_remaining:
                epochs_done = UNFREEZE_WARMUP_EPOCHS - self.group_warmup_remaining[lidx] + 1
                warmup_scale = (UNFREEZE_WARMUP_FACTOR +
                            (UNFREEZE_BUMP_FACTOR - UNFREEZE_WARMUP_FACTOR)
                            * (epochs_done / UNFREEZE_WARMUP_EPOCHS))
                group["lr"] = group["base_lr"] * cosine_scale * warmup_scale  # set, not multiply

        for k in list(self.group_warmup_remaining):
            self.group_warmup_remaining[k] -= 1
            if self.group_warmup_remaining[k] <= 0:
                del self.group_warmup_remaining[k]
   

    def restore_to_epoch(self, epoch):
        """Replay all unfreeze events that should have fired before this epoch."""
        for unfreeze_epoch in sorted(self.schedule.keys()):
            if unfreeze_epoch < epoch:
                threshold = self.schedule[unfreeze_epoch]
                for layer, idx in self.layer_to_idx.items():
                    if idx >= threshold:
                        for p in layer.parameters():
                            p.requires_grad = True
                print(f"  Restored: unfroze layers >= {threshold} (epoch {unfreeze_epoch})")

    def get_end_epoch(self):
        return max(UNFREEZE_SCHEDULE.keys())

def layer_unfreeze_epoch(layer_idx, schedule):
    if layer_idx < 0:        # head, view embed, attn_pool - always live
        return 1
    for epoch in sorted(schedule.keys()):
        if layer_idx >= schedule[epoch]:
            return epoch
    return 1


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.1)

def fit_temperature(model, val_loader, device, scaler):
    optimizer = torch.optim.LBFGS(
        scaler.parameters(), lr=0.01, max_iter=50
    )
    all_logits, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for imgs, lbls, views in val_loader:
            imgs, views = imgs.to(device), views.to(device)
            logits = model(imgs, views)
            all_logits.append(logits.cpu())
            all_labels.append(lbls)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    def eval_closure():
        optimizer.zero_grad()
        scaled = scaler(logits.to(device))
        loss = nn.functional.binary_cross_entropy_with_logits(
            scaled, labels.to(device)
        )
        loss.backward()
        return loss

    optimizer.step(eval_closure)
    print(f"Learned temperature: {scaler.temps.mean().item():.4f}")
    # no return needed - scaler is modified in place




clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# fix: re-standardize after CLAHE
def apply_clahe(imgs):
    out = []
    for img in imgs:
        gray = img[0].cpu().numpy()
        gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        gray = clahe.apply(gray)
        gray = gray.astype(np.float32) / 255.0
        gray = np.stack([gray, gray, gray], axis=0)
        t = torch.from_numpy(gray).float()
        # re-standardize to match PerImageStandardize
        t = (t - t.mean()) / (t.std() + 1e-6)
        out.append(t)
    return torch.stack(out).to(imgs.device)


def tta_predict(model, imgs, views):
    """
    Minimal chest X-ray TTA:
      - original
      - horizontal flip
      - CLAHE
      - CLAHE + flip

    Returns averaged probabilities.
    """

    clahe_imgs = apply_clahe(imgs.clone())

    aug_batches = [
        imgs,
        torch.flip(imgs, dims=[3]),
        clahe_imgs,
        torch.flip(clahe_imgs, dims=[3]),
    ]

    probs_list = []

    with torch.no_grad():

        for aug_imgs in aug_batches:

            logits = model(aug_imgs, views)

            probs = torch.sigmoid(logits)

            probs_list.append(probs)

    probs = torch.stack(probs_list, dim=0).mean(dim=0)

    return probs


def init_group_cosine(group, epoch, total_epochs, eta_min_ratio, warmup_epochs=0):
    ue = max(warmup_epochs, group.get("unfreeze_epoch", 1)) if group.get("layer_idx", -1) >= 0 else 0
    effective = max(epoch - ue, 0)
    T_max = max(total_epochs - ue, 1)
    cos = 0.5 * (1 + math.cos(math.pi * effective / T_max))
    eta_min = group["base_lr"] * eta_min_ratio
    return eta_min + (group["base_lr"] - eta_min) * cos


# Param Groups
# This controls learning rate for different parts of the model,
# and allows for gradual unfreezing of the backbone with a warmup.
def init_param_groups(model, base_lr=1e-4, decay=0.8):
    groups = []
    seen = set()

    def add(params, lr, layer_idx, weight_decay=1e-2):
        wd, no_wd = [], []
        for p in params:
            pid = id(p)
            if pid in seen: continue
            seen.add(pid)
            (no_wd if p.ndim <= 1 else wd).append(p)

        ue = layer_unfreeze_epoch(layer_idx, UNFREEZE_SCHEDULE)

        for bucket, wdv in [(wd, weight_decay), (no_wd, 0.0)]:
            if bucket:
                groups.append({
                    "params": bucket,
                    "lr": lr,
                    "base_lr": lr,
                    "layer_idx": layer_idx,
                    "unfreeze_epoch": ue,
                    "weight_decay": wdv,
                })
    

    layers = list(model.backbone.layers)

    for i, layer in enumerate(reversed(layers)):
        lr = base_lr * (decay ** i)
        layer_idx = len(layers) - 1 - i
        add(layer.parameters(), lr, layer_idx)

    add(model.backbone.patch_embed.parameters(), base_lr * (decay ** (len(layers)-1)), layer_idx=0)
    add(model.backbone.norm.parameters(), base_lr, layer_idx=-1)
    add(model.class_head.parameters(),  base_lr * HEAD_LR_MULTIPLIER, layer_idx=-1, weight_decay=0.08)
    add(
        [model.stage_gates],
        base_lr * HEAD_LR_MULTIPLIER * STAGE_GATES_MULTIPLIER,
        layer_idx=-1
    )
    add(model.stage_projs.parameters(), base_lr * HEAD_LR_MULTIPLIER * 2.0, layer_idx=-1, weight_decay=0.08)
    add(model.class_norm.parameters(),  base_lr * HEAD_LR_MULTIPLIER, layer_idx=-1, weight_decay=0.08)
    add(model.view_embed.parameters(), base_lr, -1)
    add(model.view_mlp.parameters(), base_lr, -1)
    
    add([model.view_scale, model.stage_temps],
            base_lr * HEAD_LR_MULTIPLIER, layer_idx=-1)
    add([model.class_queries], base_lr * HEAD_LR_MULTIPLIER, 
            layer_idx=-1, weight_decay=0.15)


    leftovers = [p for p in model.parameters() if id(p) not in seen]
    if leftovers:
        add(leftovers, base_lr * (decay ** len(layers)), -2)

    return groups

def print_architecture_parameters():
    print("Architecture Parameters:")
    print("  BASE_LR", BASE_LR)
    print("  BASE_LR_ADJUSTED", BASE_LR_ADJUSTED)
    print("  HEAD_LR_MULTIPLIER", HEAD_LR_MULTIPLIER)
    print("  STAGE_GATES_MULTIPLIER", STAGE_GATES_MULTIPLIER)
    print("  LR_LAYER_DECAY", LR_LAYER_DECAY)
    print("  FEATURE_DROPOUT", FEATURE_DROPOUT)
    print("  CLASSIFIER_DROPOUT",  CLASSIFIER_DROPOUT)
    print("  VIEW_POSITION_SCALE", VIEW_POSITION_SCALE)
    print("  IMAGE_SIZE", IMAGE_SIZE)
    print("  SWIN_WINDOW_SIZE", SWIN_WINDOW_SIZE)
    print("  ASYMMETRIC_CLIP", ASYMMETRIC_CLIP)
    print("  ASYMMETRIC_GAMMA_NEG", ASYMMETRIC_GAMMA_NEG)
    print("  ASYMMETRIC_GAMMA_POS", ASYMMETRIC_GAMMA_POS)
    print("  ASYMMETRIC_LABEL_SMOOTH", ASYMMETRIC_LABEL_SMOOTH)
    print("  WEIGHT_DECAY", WEIGHT_DECAY)
    print("  EMA_DECAY", EMA_DECAY)
    print("  UNFREEZE_WARMUP_EPOCHS", UNFREEZE_WARMUP_EPOCHS)
    print("  UNFREEZE_WARMUP_FACTOR", UNFREEZE_WARMUP_FACTOR)
    print("  UNFREEZE_BUMP_FACTOR", UNFREEZE_BUMP_FACTOR)
    print("  UNFREEZE_SCHEDULE", UNFREEZE_SCHEDULE)
    print("  HEAD_WARMUP_EPOCHS", HEAD_WARMUP_EPOCHS)
    print("  HEAD_WARMUP_START_FACTOR", HEAD_WARMUP_START_FACTOR)
    print("  SWA_START_EPOCH", SWA_START_EPOCH)
    print("  SWA_LR", SWA_LR)
    print("  ETA_MIN_RATIO", ETA_MIN_RATIO)
    print("  NOISY_CLASSES", NOISY_CLASSES)
    print("  PER_CLASS_GAMMA_NEG", PER_CLASS_GAMMA_NEG)
    print("  PER_CLASS_GAMMA_POS", PER_CLASS_GAMMA_POS)
    print("  PER_CLASS_CLIP", PER_CLASS_CLIP)


