"""A small frequency-spectrum baseline that runs on CPU with scikit-learn only.

It exists for two reasons. First, it lets the whole pipeline (manifest ->
train -> evaluate -> report -> serve) be exercised without a GPU. Second, it is
a sanity check on the data: if this trivial model scores close to the deep
model on a test set, that test set contains a low-level shortcut and its
numbers should not be trusted as evidence of real detection ability.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIZE = 256
N_BINS = 48


def _radial_bins(size: int, n_bins: int) -> np.ndarray:
    yy, xx = np.indices((size, size))
    r = np.hypot(yy - size // 2, xx - size // 2)
    return np.minimum((r / (size / 2) * n_bins).astype(int), n_bins - 1)


_BINS = _radial_bins(SIZE, N_BINS)
_BIN_COUNTS = np.bincount(_BINS.ravel(), minlength=N_BINS).astype(np.float64)


def spectral_features(im: Image.Image) -> np.ndarray:
    """Radially averaged log-spectrum of the high-pass residual, plus a few
    residual statistics. Generators leave traces in exactly this band."""
    short = min(im.size)
    scale = SIZE / short
    im = im.resize(
        (max(SIZE, round(im.width * scale)), max(SIZE, round(im.height * scale))),
        Image.Resampling.BILINEAR,
    )
    left, top = (im.width - SIZE) // 2, (im.height - SIZE) // 2
    gray = im.crop((left, top, left + SIZE, top + SIZE)).convert("L")
    x = np.asarray(gray, dtype=np.float32)
    smooth = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32)
    residual = x - smooth
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(residual))))
    profile = np.bincount(_BINS.ravel(), weights=spectrum.ravel(), minlength=N_BINS) / _BIN_COUNTS
    std = residual.std() + 1e-6
    extras = np.array([
        std,
        np.mean(np.abs(residual)) / std,
        np.mean((residual / std) ** 4),
        np.abs(np.diff(x, axis=0)).mean(),
        np.abs(np.diff(x, axis=1)).mean(),
    ])
    return np.concatenate([profile, extras]).astype(np.float32)


def train_baseline(
    manifest,
    out_path: str | Path,
    max_per_label: int = 5000,
    harmonize_q: int | None = 96,
    threshold: float = 0.5,
    target_fpr: float | None = None,
    workers: int = 8,
    seed: int = 0,
) -> Path:
    """Fit the baseline on the manifest's train split, calibrate on val, save a bundle."""
    from concurrent.futures import ThreadPoolExecutor

    from . import __version__
    from .calibrate import Calibrator, threshold_for_fpr
    from .evaluate import _sample
    from .imageio import ImageError
    from .scoring import prepare_image, score_paths

    train_df = _sample(manifest[manifest["split"] == "train"], max_per_label, seed)
    val_df = _sample(manifest[manifest["split"] == "val"], max_per_label, seed)
    if train_df["label"].nunique() < 2:
        raise ValueError("the train split must contain both real and generated images")

    def features(path: str):
        try:
            return spectral_features(prepare_image(path, harmonize_q))
        except ImageError:
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        feats = list(pool.map(features, train_df["path"]))
    keep = [i for i, f in enumerate(feats) if f is not None]
    x = np.stack([feats[i] for i in keep])
    y = train_df["label"].to_numpy()[keep]
    baseline = SpectralBaseline().fit(x, y)

    calibrator = Calibrator()
    if len(val_df):
        val_logits = score_paths(list(val_df["path"]), baseline, harmonize_q, workers=workers)
        val_y = val_df["label"].to_numpy()
        calibrator = Calibrator.fit(val_logits, val_y)
        if target_fpr:
            threshold = threshold_for_fpr(calibrator(val_logits[val_y == 0]), float(target_fpr))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.save(
        out_path,
        calibration=calibrator.to_dict(),
        threshold=float(threshold),
        harmonize_q=harmonize_q,
        meta={
            "model_name": SpectralBaseline.name,
            "aidetect_version": __version__,
            "trained_on_generators": sorted(train_df.loc[train_df["label"] == 1, "generator"].unique().tolist()),
            "n_train": int(len(keep)),
        },
    )
    return out_path


class SpectralBaseline:
    name = "spectral_baseline"

    def __init__(self, model=None):
        self.model = model

    def fit(self, images_or_features, y) -> "SpectralBaseline":
        feats = self._features(images_or_features)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        self.model.fit(feats, np.asarray(y).astype(int))
        return self

    @staticmethod
    def _features(items) -> np.ndarray:
        if isinstance(items, np.ndarray):
            return items
        return np.stack([spectral_features(im) for im in items])

    def score(self, images: list[Image.Image]) -> np.ndarray:
        return self.model.decision_function(self._features(images))

    def save(self, path: str | Path, **payload) -> None:
        joblib.dump({"kind": self.name, "model": self.model, **payload}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["SpectralBaseline", dict]:
        payload = joblib.load(path)
        if payload.get("kind") != cls.name:
            raise ValueError(f"{path} is not a {cls.name} bundle")
        return cls(payload.pop("model")), payload
