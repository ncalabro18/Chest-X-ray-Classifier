import numpy as np
import pandas as pd
import math

from dataset import ALL_CLASSES
import torch.nn as nn
import torch

from sklearn.model_selection import StratifiedShuffleSplit

### Architecture Parameters
# not required in train_save.py
# but should be logged

FEATURE_DROPOUT    = 0.2
CLASSIFIER_DROPOUT = 0.1

HEAD_LR_MULTIPLIER = 6
VIEW_POSITION_SCALE = 0.2


# Check point file labels that aren't required
EXPECTED_MISSING = {
    "relative_coords_table",
    "relative_position_index",
    "attn_mask"
}


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
        # self.stage_projs = nn.ModuleList([
        #     nn.Linear(d, C) if d != C else nn.Identity()
        #     for d in stage_dims
        # ])
        self.stage_projs = nn.ModuleList([nn.Conv2d(d, C, 1) for d in stage_dims[:-1]])



        self.class_queries = nn.Parameter(torch.randn(num_classes, C) * 0.02)
        self.class_norm = nn.LayerNorm(C)
        self.attn_scale = C ** -0.5
        
        self.view_embed = torch.nn.Embedding(2, 32)
        self.view_mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, C * 2)
        )
        self.view_scale = torch.nn.Parameter(torch.tensor(VIEW_POSITION_SCALE))

        # Init

        nn.init.normal_(self.view_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.view_mlp[-1].bias)
        nn.init.trunc_normal_(self.class_queries, std=0.02)

        for proj in self.stage_projs:
            if isinstance(proj, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(proj.weight.view(proj.weight.size(0), -1)
                                        if isinstance(proj, nn.Conv2d) else proj.weight)
                nn.init.zeros_(proj.bias)

        self.stage_temps = nn.Parameter(torch.ones(len(self.backbone.layers)))


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
        for feat, proj in zip(early_feats, self.stage_projs):
            B, N, D = feat.shape
            h = w = int(N ** 0.5)
            feat_2d = feat.reshape(B, h, w, D).permute(0, 3, 1, 2)
            projected = proj(feat_2d).flatten(2).transpose(1, 2)
            projected = self.class_norm(projected)                 # mirror forward()
            stage_tokens.append(projected)

        x_normed = self.backbone.norm(x)
        x_normed = self.class_norm(x_normed)
        N_final = x_normed.shape[1]

        # Apply stage_temps exactly as in forward()
        scaled_tokens = []
        for i, tokens in enumerate(stage_tokens):
            scaled_tokens.append(tokens * self.stage_temps[i].abs())
        scaled_tokens.append(x_normed * self.stage_temps[-1].abs())
        all_tokens = torch.cat(scaled_tokens, dim=1)
        
        
        # Class query cross-attention over all scales simultaneously
        B = all_tokens.size(0)
        Q = self.class_queries.unsqueeze(0).expand(B, -1, -1)       # (B, num_classes, C)
        attn = torch.bmm(Q, all_tokens.transpose(1, 2)) * self.attn_scale  # (B, num_classes, N_total)
        attn = torch.softmax(attn, dim=-1)
        class_feats = torch.bmm(attn, all_tokens)                    # (B, num_classes, C)

        # View conditioning
        v = self.view_mlp(self.view_embed(view_id))                  # (B, C*2)
        gamma, beta = v.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        scale = torch.sigmoid(self.view_scale) * 2.0
        class_feats = class_feats * (1 + scale * gamma.unsqueeze(1)) + beta.unsqueeze(1)

        logits = self.class_head(class_feats).squeeze(-1)            # (B, num_classes)
        return logits    



# Loss
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, clip=0.05,
                 eps=1e-8,  label_smooth=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.label_smooth = label_smooth


    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Clip negative probabilities
        if self.clip > 0:
            probs_neg = (1 - probs - self.clip).clamp(min=0)
        else:
            probs_neg = 1 - probs

        # Asymmetric focusing
        pos_focal = (1 - probs) ** self.gamma_pos
        neg_focal = probs ** self.gamma_neg

        if self.label_smooth > 0:
                    targets = targets * (1 - self.label_smooth) + 0.5 * self.label_smooth

        # Loss
        loss_pos = targets * torch.log(probs.clamp(min=self.eps)) * pos_focal
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps)) * neg_focal

        loss = -(loss_pos + loss_neg).mean()
        return loss


class UnfreezeScheduler:
    def __init__(self, layer_to_idx, optimizer, schedule, warmup_epochs):
        self.epoch = 1
        self.optimizer = optimizer
        self.schedule = schedule
        self.layer_to_idx = layer_to_idx
        self.warmup_epochs = warmup_epochs
        # freeze layers
        for layer, idx in layer_to_idx.items():
            for p in layer.parameters():
                p.requires_grad = False

    # Unfreeze layers per schedule
    def step(self, group_warmup_remaining):
        newly_unfrozen = set()
        if self.epoch in self.schedule:
            if group_warmup_remaining:
                print(f"WARNING: Unfreezing at epoch {self.epoch} but warmup still active for layers: {list(group_warmup_remaining.keys())}")
            
            threshold = self.schedule[self.epoch]
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
                    group_warmup_remaining[lidx] = self.warmup_epochs
        self.epoch += 1

def layer_unfreeze_epoch(layer_idx, schedule):
    if layer_idx < 0:        # head, view embed, attn_pool — always live
        return 1
    for epoch in sorted(schedule.keys()):
        if layer_idx >= schedule[epoch]:
            return epoch
    return 1


# Extracts labels and view_ids, and filters to PA/AP images only
def init_metadata(path):
    df = pd.read_csv(path)
    df = df[["Image Index", "Finding Labels",
             "View Position", "Patient ID"]].copy()
    df["view_id"] = df["View Position"].map({"PA": 0, "AP": 1}).fillna(0).astype(int)
    df = df[df["View Position"].isin(["PA", "AP"])].reset_index(drop=True)
    df["labels"] = df["Finding Labels"].str.split("|")
    
    return df

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


def init_group_cosine(group, epoch, total_epochs, eta_min_ratio, warmup_epochs):
    ue = warmup_epochs if group.get("layer_idx", -1) < 0 else group.get("unfreeze_epoch", 1)
    effective = max(epoch - ue, 0)
    T_max = max(total_epochs - ue, 1)
    cos = 0.5 * (1 + math.cos(math.pi * effective / T_max))
    eta_min = group["base_lr"] * eta_min_ratio # per-group floor
    return eta_min + (group["base_lr"] - eta_min) * cos


def init_split(df, label_matrix):
    patient_ids = df["Patient ID"].unique()

    patient_label_matrix = np.zeros((len(patient_ids), len(ALL_CLASSES)), dtype=int)
    patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    for img_idx, row in df.iterrows():
        p = patient_id_to_idx[row["Patient ID"]]
        patient_label_matrix[p] |= label_matrix[img_idx]

    # Collapse multilabel to a single stratification key via label combination hash
    # Rare combos get lumped into a single "other" bin to avoid singleton strata
    combo_strings = ["_".join(map(str, row)) for row in patient_label_matrix]
    from collections import Counter
    counts = Counter(combo_strings)
    MIN_COMBO_COUNT = 2
    strat_labels = [c if counts[c] >= MIN_COMBO_COUNT else "__other__" for c in combo_strings]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_patient_idx, val_patient_idx = next(sss.split(patient_ids, strat_labels))

    train_patients = set(patient_ids[train_patient_idx])
    value_patients = set(patient_ids[val_patient_idx])

    train_idx = df[df["Patient ID"].isin(train_patients)].index.to_numpy()
    value_idx = df[df["Patient ID"].isin(value_patients)].index.to_numpy()

    for split_name, idx in [("train", train_idx), ("val", value_idx)]:
        n_hernia = label_matrix[idx, ALL_CLASSES.index("Hernia")].sum()
        print(f"{split_name} Hernia positives: {n_hernia}")

    return train_idx, value_idx

# Loads the SSL checkpoint from the path SSL_CKPT
def init_ckpt(model, path):
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


# Param Groups
# This controled learning rate for different parts of the model,
# and allows for gradual unfreezing of the backbone with a warmup.
def init_param_groups(model, base_lr=1e-4, decay=0.8, schedule=None):
    schedule = schedule or {}
    groups = []
    seen = set()

    def add(params, lr, layer_idx, weight_decay=1e-2):
        wd, no_wd = [], []
        for p in params:
            pid = id(p)
            if pid in seen: continue
            seen.add(pid)
            (no_wd if p.ndim <= 1 else wd).append(p)

        ue = layer_unfreeze_epoch(layer_idx, schedule)

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
    add(model.class_head.parameters(), base_lr * HEAD_LR_MULTIPLIER, layer_idx=-1)
    add(model.stage_projs.parameters(), base_lr * HEAD_LR_MULTIPLIER, layer_idx=-1)
    add(model.view_embed.parameters(), base_lr, -1)
    add(model.view_mlp.parameters(), base_lr, -1)
    add(model.class_norm.parameters(), base_lr, layer_idx=-1)
    add([model.class_queries, model.view_scale, model.stage_temps],
            base_lr * HEAD_LR_MULTIPLIER, layer_idx=-1)


    leftovers = [p for p in model.parameters() if id(p) not in seen]
    if leftovers:
        add(leftovers, base_lr * (decay ** len(layers)), -2)

    return groups

def print_architecture_parameters():
    print("Architecture parameters:")
    print(f"  FEATURE_DROPOUT: {FEATURE_DROPOUT}")
    print(f"  CLASSIFIER_DROPOUT: {CLASSIFIER_DROPOUT}")
    print(f"  HEAD_LR_MULTIPLIER: {HEAD_LR_MULTIPLIER}")
    print(f"  VIEW_POSITION_SCALE: {VIEW_POSITION_SCALE}")
