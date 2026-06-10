"""
© 2026 Nicholas J. Calabro. All rights reserved.

Accepts a POST request at /submit and forwards it to the classifier service.

Uses an asyncio.Semaphore to limit concurrent submissions.
If all slots are occupied, returns HTTP 429.
"""

import asyncio
import logging
import os
import time
import uuid
import asyncio
import json
from fastapi.responses import JSONResponse
import httpx


from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from fastapi import (
    FastAPI, Request, File, Form, HTTPException, UploadFile
)
from PIL import Image
from io import BytesIO
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import start_http_server
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_BYTES = int(os.getenv("MAX_BYTES", str(10 * 1024 * 1024)))
MAX_WIDTH = 1080
MAX_HEIGHT = 1080
MAX_CONCURRENT_REQUESTS = 5
MAX_ASPECT_RATIO = 4.0
MAX_IMAGE_PIXELS = 20_000_000

CLASSIFIER_BASE_URL = os.getenv("CLASSIFIER_URL")
CLASSIFIER_API_KEY = os.getenv("CLASSIFIER_API_KEY")



class TimeoutMiddleware:
    def __init__(self, app, timeout: float = 65.0):
        self.app = app
        self.timeout = timeout

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            await asyncio.wait_for(
                self.app(scope, receive, send),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            response = JSONResponse(
                {"detail": "Request timeout"}, status_code=408
            )
            await response(scope, receive, send)




if CLASSIFIER_API_KEY is None:
    print("ERROR, NO CLASSIFIER API KEY FOUND;")
    print("Quitting...")
    exit(-1)

# Metrics
submissions_total = Counter(
    'cxr_web_submissions_total',
    'Total image submissions by outcome',
    ['view', 'outcome'],   # outcome: success | rejected_mime | rejected_size
                           #    rejected_dims | invalid_image | busy | error
)

end_to_end_latency = Histogram(
    'cxr_web_request_seconds',
    'Total request latency from receipt to response (includes classifier roundtrip)',
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0],
)

classifier_roundtrip = Histogram(
    'cxr_web_classifier_roundtrip_seconds',
    'Time spent waiting for the classifier service to respond',
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0],
)

semaphore_slots_free = Gauge(
    'cxr_web_semaphore_slots_free',
    'Number of currently available semaphore slots',
)

classifier_status_total = Counter(
    'cxr_web_classifier_status_total',
    'Classifier HTTP response codes seen by web service',
    ['status_code'],
)

image_size_bytes = Histogram(
    'cxr_web_image_bytes',
    'Size of accepted images in bytes',
    buckets=[50_000, 100_000, 250_000, 500_000, 1_000_000,
             2_000_000, 5_000_000, 10_000_000],
)


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_http_server(port=9091)
    yield

# App
app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(TimeoutMiddleware, timeout=65.0)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

Instrumentator().instrument(app)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


async def forward_to_classifier(view: str, img_data: bytes) -> tuple[int, str]:
    filename = f"{uuid.uuid4()}.png"
    timeout = httpx.Timeout(
        connect=10.0,
        read=55.0,
        write=30.0,
        pool=10.0,
    )
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()
        r = await client.post(
            CLASSIFIER_BASE_URL + "/image",
            data={"view": view},
            files={"image": (filename, img_data, "image/png")},
            headers={
                "X-API-Key": CLASSIFIER_API_KEY
            }
        )
        classifier_roundtrip.observe(time.perf_counter() - t0)

    classifier_status_total.labels(status_code=str(r.status_code)).inc()
    logger.info("Classifier response: %d", r.status_code)
    return r.status_code, r.text



@app.post("/submit")
@limiter.limit("6/minute")
async def submit_image(
    request: Request,
    view: str = Form(...),
    file: UploadFile = File(...),
):
    t_start = time.perf_counter()
    logger.info( # don't log filename; security concern
        "submit/ POST RECEIVED",
    )
    # secret check set in caddy
    if request.headers.get("X-Tunnel-Secret") != os.getenv("INTERNAL_SECRET"):
        raise HTTPException(status_code=403, detail="Invalid header secret")

        
    # view check
    if view not in {"AP", "PA"}:
        submissions_total.labels(view=view, outcome="rejected_view").inc()
        raise HTTPException(status_code=400, detail="View must be AP or PA")
    
    # content type check
    if not file.content_type.startswith("image/"):
        submissions_total.labels(view=view, outcome="rejected_mime").inc()
        raise HTTPException(
            status_code=415,
            detail="Uploaded file must be an image"
        )
    
    # png check
    if file.content_type != "image/png":
        submissions_total.labels(view=view, outcome="rejected_mime").inc()
        raise HTTPException(status_code=415, detail="Only PNG images are allowed")

    # Semaphore - non-blocking check with short timeout
    acquired = False
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.05)
        acquired = True
        semaphore_slots_free.set(semaphore._value)
    except asyncio.TimeoutError:
        submissions_total.labels(view=view, outcome="busy").inc()
        raise HTTPException(
            status_code=429,
            detail="Server busy; try again later"
        )

    try:

        # Size check
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            submissions_total.labels(view=view, outcome="rejected_size").inc()
            raise HTTPException(status_code=413, detail="File too large")

        image_size_bytes.observe(len(data))

        
        # Decode and validate
        try:
            img = Image.open(BytesIO(data))
            img.load()
        except Exception as e:
            logger.warning("Invalid image data: %s", e)
            submissions_total.labels(view=view, outcome="invalid_image").inc()
            raise HTTPException(status_code=400, detail="Invalid image data")

        if img.format != "PNG":
            submissions_total.labels(view=view, outcome="invalid_image").inc()
            raise HTTPException(status_code=400, detail="Invalid PNG file")

        if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
            submissions_total.labels(view=view, outcome="rejected_dims").inc()
            raise HTTPException(
                status_code=400,
                detail="Image dimensions too large"
            )

        # aspect ratio check
        ratio = max(img.width / img.height, img.height / img.width)
        if ratio > MAX_ASPECT_RATIO:
            raise HTTPException(
                status_code=400,
                detail="Invalid image dimensions"
            )
        
        # Sanitise by re-encoding as greyscale PNG
        img = img.convert("L")
        clean_buf = BytesIO()
        img.save(clean_buf, format="PNG", optimize=False)
        clean_data = clean_buf.getvalue()



        # Forward to classifier
        status, resp_text = await forward_to_classifier(view, clean_data)

        if status != 200:
            raise HTTPException(
                status_code=502,
                detail="Classifier unavailable"
            )

        outcome = "success" if status == 200 else "error"
        submissions_total.labels(view=view, outcome=outcome).inc()
        end_to_end_latency.observe(time.perf_counter() - t_start)

        parsed = json.loads(resp_text)

        return {
            "status": "forwarded",
            "view": view,
            "classifier_status": status,
            "classifier_response": parsed,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        submissions_total.labels(view=view, outcome="error").inc()
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        if acquired:
            semaphore.release()
            semaphore_slots_free.set(semaphore._value)


@app.get("/status")
@limiter.limit("14/minute")
async def status(
    request: Request,
):

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                f"{CLASSIFIER_BASE_URL}/ready",
                headers={
                    "X-API-Key": CLASSIFIER_API_KEY
                }
            )
            model_ready = r.status_code == 200
    except Exception:
        model_ready = False

    state = None

    if not model_ready:
        state = "starting"
    elif semaphore._value == 0:
        state = "busy"
    else:
        state = "ready"

    return {"state": state, "slots_free": semaphore._value}