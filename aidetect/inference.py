"""The inference module: load a bundle, classify one image.

    from aidetect.inference import ImageDetector
    detector = ImageDetector.load("runs/main/model.pt")
    result = detector.predict("photo.jpg")
    print(result.label, result.confidence, result.ai_probability)
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

from PIL import Image

from . import LABEL_NAMES
from .calibrate import Calibrator
from .imageio import ImageSource, harmonize, open_image
from .scoring import Scorer, load_bundle


@dataclass
class Prediction:
    label: str              # "Real" or "AI-Generated"
    confidence: float       # probability assigned to the returned label, 0..1
    ai_probability: float   # calibrated probability the image is AI-generated, 0..1
    threshold: float        # ai_probability at or above this gives "AI-Generated"
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


class ImageDetector:
    def __init__(
        self,
        scorer: Scorer,
        calibrator: Calibrator | None = None,
        threshold: float = 0.5,
        harmonize_q: int | None = 96,
        model_name: str | None = None,
    ):
        self.scorer = scorer
        self.calibrator = calibrator or Calibrator()
        self.threshold = float(threshold)
        self.harmonize_q = harmonize_q
        self.model_name = model_name or getattr(scorer, "name", "detector")
        self._lock = threading.Lock()  # one forward pass at a time per model

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "ImageDetector":
        b = load_bundle(path, device=device)
        name = b["meta"].get("model_name") or getattr(b["scorer"], "name", "detector")
        return cls(b["scorer"], b["calibrator"], b["threshold"], b["harmonize_q"], name)

    def _prediction(self, ai_probability: float) -> Prediction:
        p = min(max(float(ai_probability), 0.0), 1.0)
        is_ai = p >= self.threshold
        return Prediction(
            label=LABEL_NAMES[int(is_ai)],
            confidence=p if is_ai else 1.0 - p,
            ai_probability=p,
            threshold=self.threshold,
            model=self.model_name,
        )

    def predict_images(self, images: list[Image.Image], formats: list[str]) -> list[Prediction]:
        """Classify already-decoded RGB images; `formats` are their source formats."""
        prepared = [harmonize(im, fmt, self.harmonize_q) for im, fmt in zip(images, formats)]
        with self._lock:
            logits = self.scorer.score(prepared)
        return [self._prediction(p) for p in self.calibrator(logits)]

    def predict(self, src: ImageSource) -> Prediction:
        """Classify one image given as a file path or raw bytes.
        Raises `aidetect.imageio.ImageError` if it cannot be decoded."""
        im, fmt = open_image(src)
        return self.predict_images([im], [fmt])[0]
