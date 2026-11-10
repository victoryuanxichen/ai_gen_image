"""HTTP inference service.

    AIDETECT_MODEL=runs/main/model.pt uvicorn aidetect.serve:app --host 0.0.0.0 --port 8000

    curl -F "file=@photo.jpg" http://localhost:8000/v1/detect
    {"label":"Real","confidence":0.97,"ai_probability":0.03,"threshold":0.5,"model":"..."}

Needs fastapi, python-multipart and uvicorn.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .imageio import ImageError, open_image
from .inference import ImageDetector

ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_BYTES = int(os.environ.get("AIDETECT_MAX_BYTES", 20 * 1024 * 1024))


class DetectResponse(BaseModel):
    label: str = Field(description='"Real" or "AI-Generated"')
    confidence: float = Field(ge=0, le=1, description="Probability assigned to the returned label")
    ai_probability: float = Field(ge=0, le=1, description="Calibrated probability the image is AI-generated")
    threshold: float = Field(description="ai_probability at or above this is labelled AI-Generated")
    model: str


def create_app(detector: ImageDetector | None = None) -> FastAPI:
    """Build the app. Pass a detector directly (tests) or set AIDETECT_MODEL."""
    state: dict = {"detector": detector}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["detector"] is None:
            path = os.environ.get("AIDETECT_MODEL")
            if not path:
                raise RuntimeError("set AIDETECT_MODEL to the path of a trained model bundle")
            state["detector"] = ImageDetector.load(path, device=os.environ.get("AIDETECT_DEVICE"))
        yield

    app = FastAPI(title="AI-generated image detector", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        det = state["detector"]
        return {"status": "ok" if det is not None else "loading", "model": getattr(det, "model_name", None)}

    # A plain `def` endpoint runs in FastAPI's thread pool, so decoding and the
    # forward pass do not block the event loop.
    @app.post("/v1/detect", response_model=DetectResponse)
    def detect(file: UploadFile = File(...)) -> DetectResponse:
        data = file.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"file larger than {MAX_BYTES} bytes")
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        try:
            image, fmt = open_image(data)
        except ImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if fmt not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415, detail=f"unsupported image format '{fmt}'; send JPEG or PNG"
            )
        prediction = state["detector"].predict_images([image], [fmt])[0]
        return DetectResponse(**prediction.to_dict())

    return app


app = create_app()
