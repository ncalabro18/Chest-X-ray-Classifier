"""
© 2026 Nicholas J. Calabro. All rights reserved.


Extra Utility Functions
- init_metadata: Loads and processes the metadata CSV,
        extracting labels and view positions
- init_ckpt: Loads a checkpoint file,
        handling both training and SSL formats
- init_device: Sets up the PyTorch device,
        preferring GPU if available,
        and prints device info

"""
import torch
import pandas as pd

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
