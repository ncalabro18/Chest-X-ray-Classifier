"""
© 2026 Nicholas J. Calabro. All rights reserved.
classifier.py
Determines diseases from a POST request
Responds with per-disease probabilities, thresholds,
GradCAM, and Attention Map
"""
import io
import threading
import time
import numpy as np
import torch
import uvicorn
import secrets
import os
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware

from classes import ALL_CLASSES, NUM_CLASSES
from dataset import PerImageStandardize, make_value_tf
from architecture import MultiClassifier, IMAGE_SIZE, tta_predict
from visualization_util import generate_attention_map, generate_gradient_saliency
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import (Counter, Histogram, start_http_server)
from contextlib import asynccontextmanager


MODEL_OUTPUT_FILE = "swin_cxr8_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_val_tf = make_value_tf(IMAGE_SIZE)

API_KEY = os.environ["API_KEY"]

MAX_IMAGE_BYTES = 10 * 1024 * 1024


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

INFERENCE_TOTAL = Counter(
    'classifier_inferences_total',
    'Total inference requests by outcome',
    ['view', 'outcome'],  # outcome: success | error
)
INFERENCE_LATENCY = Histogram(
    'classifier_inference_duration_seconds',
    'Model inference wall-clock time',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
MAP_GENERATION_LATENCY = Histogram(
       'classifier_map_generation_seconds',
    'Model inference wall-clock time',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
ATTENTION_MAP_ERROR =  Counter(
    'classifier_attnmap_errors',
    'Total saliency generation errors',
    ['view'],
)
SALIENCY_MAP_ERROR =  Counter(
    'classifier_saliencymap_errors',
    'Total saliency generation errors',
    ['view'],
)

# Startup app
startup_app = FastAPI()
startup_app.add_middleware(APIKeyMiddleware)

@startup_app.get("/ping")
async def startup_ping():
    return {"message": "hello"}

@startup_app.get("/ready")
async def startup_ready():
    raise HTTPException(status_code=503, detail="Model loading")



@asynccontextmanager
async def lifespan(app: FastAPI):
    start_http_server(port=9091)
    yield

# Main app
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.add_middleware(APIKeyMiddleware)




@app.get("/ping")
@limiter.limit("10/minute")
async def ping(request: Request):
    return {"message": "hello"}

@app.get("/ready")
@limiter.limit("10/minute")
async def ready(request: Request):
    return {"status": "ready"}

@app.post("/image")
@limiter.limit("6/minute")
async def classify_image(
    request: Request,
    image: UploadFile = File(...),
    view: str = Form(default="PA"),
):
    
    print("Classifying a ", view, " image.")
    print("  filename = ", image.filename)
    print("  size = ", image.size)
    print("  content_type = ", image.content_type)
    view = view.strip().upper()
    if view not in ("PA", "AP"):
        raise HTTPException(status_code=400, detail="view must be 'PA' or 'AP'")
    view_id = 0 if view == "PA" else 1

    raw = await image.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    
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

    try:
        t0 = time.perf_counter()
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

        # Single forward pass caches last_attention for ALL classes
        with torch.no_grad():
            inner_model = model.module if hasattr(model, "module") else model
            inner_model(tensor, view_tensor)
        INFERENCE_LATENCY.observe(time.perf_counter() - t0)
        INFERENCE_TOTAL.labels(view=view, outcome='success').inc()
    except Exception:
        INFERENCE_TOTAL.labels(view=view, outcome='error').inc()
        raise

    attention_maps: dict[str, str] = {}
    saliency_maps:  dict[str, str] = {}

    MAX_VIZ_CLASSES = 3
    positive_classes = sorted(
        [(c, cls_name) for c, cls_name in enumerate(ALL_CLASSES)
         if predictions[cls_name]["positive"]],
        key=lambda x: predictions[x[1]]["probability"],
        reverse=True
    )[:MAX_VIZ_CLASSES]

    t0 = time.perf_counter()
    for c, cls_name in positive_classes:

        try:
            # Pass skip_forward=True so it uses the cached last_attention
            attention_maps[cls_name] = generate_attention_map(
                model=model,
                input_tensor=tensor,
                view_tensor=view_tensor,
                original_image=pil_img,
                class_idx=c,
                skip_forward=True,     # ADD this flag
            )
        except Exception as exc:
            ATTENTION_MAP_ERROR.labels(view=view).inc()
            logging.warning("Attention map failed for %s: %s", cls_name, exc)

        try:
            saliency_maps[cls_name] = generate_gradient_saliency(
                model=model,
                input_tensor=tensor,
                view_tensor=view_tensor,
                original_image=pil_img,
                class_idx=c,
            )
        except Exception as exc:
            SALIENCY_MAP_ERROR.labels(view=view).inc()
            logging.warning("Saliency map failed for %s: %s", cls_name, exc)

    INFERENCE_LATENCY.observe(time.perf_counter() - t0)
    INFERENCE_TOTAL.labels(view=view, outcome='success').inc()

    return {
        "predictions": predictions,
        "view": view,
        "attention_maps": attention_maps,
        "saliency_maps":  saliency_maps,
    }


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
