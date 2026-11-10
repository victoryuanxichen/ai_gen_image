"""The one scoring path used by training validation, evaluation and serving.

A *scorer* turns RGB PIL images into raw real-vs-generated scores (logits,
higher = more likely AI-generated). Two implementations exist: the deep model
(`torch_model.TorchScorer`) and a small spectral baseline
(`baseline.SpectralBaseline`) that needs no GPU.

A *bundle* is a scorer saved together with its calibration, decision threshold
and preprocessing settings, so a model file fully determines service behaviour.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from . import perturb
from .calibrate import Calibrator
from .imageio import ImageError, ImageSource, harmonize, open_image


class Scorer(Protocol):
    name: str

    def score(self, images: list[Image.Image]) -> np.ndarray:
        """Return one raw score per image."""
        ...


def prepare_image(
    src: ImageSource,
    harmonize_q: int | None = 96,
    perturbation: str = perturb.CLEAN,
    seed: int = 0,
    key: str | None = None,
) -> Image.Image:
    """Decode, optionally perturb, then harmonize the compression history.

    A perturbed image is treated as if it had been saved again: as a JPEG when
    the perturbation ends with JPEG compression, otherwise as a lossless file.
    The harmonization step then applies to it exactly as the service would.
    """
    im, fmt = open_image(src)
    if perturbation and perturbation != perturb.CLEAN:
        rng = perturb.rng_for(key if key is not None else str(src)[:200], perturbation, seed)
        im = perturb.apply(perturbation, im, rng)
        fmt = "JPEG" if perturb.ends_with_jpeg(perturbation) else "PNG"
    return harmonize(im, fmt, harmonize_q)


def score_paths(
    paths,
    scorer: Scorer,
    harmonize_q: int | None = 96,
    perturbation: str = perturb.CLEAN,
    batch_size: int = 64,
    workers: int = 8,
    seed: int = 0,
    progress: bool = False,
) -> np.ndarray:
    """Score image files. Unreadable files get NaN instead of stopping the run."""
    paths = [str(p) for p in paths]
    out = np.full(len(paths), np.nan, dtype=np.float64)

    def load(path: str):
        try:
            return prepare_image(path, harmonize_q, perturbation, seed, key=path)
        except ImageError:
            return None

    iterator = range(0, len(paths), batch_size)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"scoring [{perturbation}]", unit="batch")
        except ImportError:
            pass
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in iterator:
            chunk = paths[start : start + batch_size]
            images = list(pool.map(load, chunk))
            ok = [i for i, im in enumerate(images) if im is not None]
            if ok:
                scores = np.asarray(scorer.score([images[i] for i in ok]), dtype=np.float64)
                out[[start + i for i in ok]] = scores
    return out


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def load_bundle(path: str | Path, device: str | None = None) -> dict:
    """Load a saved model. Returns a dict with keys
    ``scorer, calibrator, threshold, harmonize_q, meta``."""
    path = Path(path)
    if path.suffix == ".joblib":
        from .baseline import SpectralBaseline

        scorer, payload = SpectralBaseline.load(path)
    elif path.suffix in (".pt", ".pth"):
        from .torch_model import TorchScorer  # imported lazily: needs PyTorch

        scorer, payload = TorchScorer.load(path, device=device)
    else:
        raise ValueError(f"unknown bundle type '{path.suffix}' (expected .pt or .joblib)")
    return {
        "scorer": scorer,
        "calibrator": Calibrator.from_dict(payload.get("calibration")),
        "threshold": float(payload.get("threshold", 0.5)),
        "harmonize_q": payload.get("harmonize_q", 96),
        "meta": payload.get("meta", {}),
    }
