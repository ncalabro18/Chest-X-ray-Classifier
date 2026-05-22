"""
© 2026 Nicholas J. Calabro. All rights reserved.
classifier.py
Determines diseases from a POST request
Responds with per-disease probabilities, thresholds,
GradCAM, and Attention Map
"""
import io
import threading
import numpy as np
import torch
import uvicorn
import secrets
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware

from classes import ALL_CLASSES, NUM_CLASSES
from dataset import PerImageStandardize, make_value_tf
from architecture import MultiClassifier, IMAGE_SIZE, tta_predict
from visualization_util import generate_attention_map, generate_grad_cam

MODEL_OUTPUT_FILE = "swin_cxr8_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_val_tf = make_value_tf(IMAGE_SIZE)

API_KEY = os.environ["API_KEY"]

class APIKeyMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):

        key = request.headers.get("X-API-Key")

        if not secrets.compare_digest(key or "", API_KEY):
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401
            )

        return await call_next(request)

state: dict = {}


# Startup app
startup_app = FastAPI()
startup_app.add_middleware(APIKeyMiddleware)

@startup_app.get("/ping")
async def startup_ping():
    return {"message": "hello"}

@startup_app.get("/ready")
async def startup_ready():
    raise HTTPException(status_code=503, detail="Model loading")


# Main app
app = FastAPI()
app.add_middleware(APIKeyMiddleware)




@app.get("/ping")
async def ping():
    return {"message": "hello"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/image")
async def classify_image(
    image: UploadFile = File(...),
    view: str = Form(default="PA"),
):
    view = view.strip().upper()
    if view not in ("PA", "AP"):
        raise HTTPException(status_code=400, detail="view must be 'PA' or 'AP'")
    view_id = 0 if view == "PA" else 1

    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        tensor = preprocess(pil_img).to(DEVICE)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {exc}")

    view_tensor = torch.tensor([view_id], dtype=torch.long, device=DEVICE)
    model      = state["model"]
    scaler     = state["scaler"]
    thresholds = state["thresholds"]

    with torch.no_grad():
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                            enabled=DEVICE.type == "cuda"):
            probs = tta_predict(model, tensor, view_tensor)
        if scaler is not None:
            logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
            probs  = torch.sigmoid(scaler(logits))

    probs_np = probs.squeeze(0).float().cpu().numpy()

    predictions = {
        cls_name: {
            "probability": round(float(probs_np[c]), 4),
            "threshold":   round(float(thresholds[c]), 4),
            "positive":    bool(probs_np[c] >= thresholds[c]),
        }
        for c, cls_name in enumerate(ALL_CLASSES)
    }

    view_tensor = torch.tensor([view_id], dtype=torch.long, device=DEVICE)

    input_tensor = tensor.clone().detach().requires_grad_(True)

    attention_map = generate_attention_map(
        model=model,
        input_tensor=input_tensor,
        view_tensor=view_tensor,
        original_image=pil_img
    )

    grad_cam = generate_grad_cam(
        model=model,
        input_tensor=input_tensor,
        view_tensor=view_tensor,
        original_image=pil_img
    )

    return {
        "predictions": predictions,
        "view": view,
        "attention_map": attention_map,
        "grad_cam": grad_cam,
    }
    # return {"predictions": predictions, "view": view}



# Helpers
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("RGB")
    arr = np.array(img)
    tensor = _val_tf(image=arr)["image"]
    tensor = tensor.float() / 255.0
    tensor = PerImageStandardize()(tensor)
    return tensor.unsqueeze(0)

def load_model():
    dummy_labels = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    dummy_idx    = np.array([0])

    classifier = MultiClassifier(
        backbone_path=MODEL_OUTPUT_FILE,
        device=DEVICE,
        label_matrix=dummy_labels,
        train_idx=dummy_idx,
    )
    classifier.load_best_checkpoint(MODEL_OUTPUT_FILE)
    print(f"  view_scale: {classifier.view_scale():.4f}")

    inference_model = classifier.ema_model.module
    inference_model.eval()
    state["model"]      = inference_model
    state["thresholds"] = classifier.load_thresholds(MODEL_OUTPUT_FILE)

    if classifier.temperature_scaler is not None:
        state["scaler"] = classifier.temperature_scaler
        print(f"  Temperature scaler loaded (mean temp: "
              f"{classifier.temperature_scaler.temps.mean().item():.4f})")
    else:
        state["scaler"] = None
        print("  No temperature scaler found; skipping calibration.")

    print("Model ready.")

# Entry point
# startup_server provides endpoints to retrieve server status
if __name__ == "__main__":
    print("Nicholas J. Calabro")
    print("Lung X-ray Multi-Classifier")
    print("Initializing startup server")
    startup_server = uvicorn.Server(uvicorn.Config(
        startup_app, host="0.0.0.0", port=9000, log_level="warning"
    ))
    t = threading.Thread(target=startup_server.run, daemon=True)
    t.start()

    print("Initializing Model")

    load_model()

    startup_server.should_exit = True
    t.join()

    uvicorn.run(app, host="0.0.0.0", port=9000)
