"""Image perturbations for robustness tests and training augmentation.

A perturbation is written as a spec string such as ``jpeg:q=75`` or
``resize:scale=0.5+jpeg:q=75`` (operations chained left to right with ``+``).
``clean`` means no change. All operations take and return RGB PIL images and
are deterministic given the random generator passed in.
"""

from __future__ import annotations

import io
import zlib
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from .imageio import encode_jpeg

CLEAN = "clean"

_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def jpeg(im: Image.Image, rng: np.random.Generator, q: float = 75) -> Image.Image:
    return encode_jpeg(im, int(round(q)))


def webp(im: Image.Image, rng: np.random.Generator, q: float = 75) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=int(round(q)))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def resize(
    im: Image.Image, rng: np.random.Generator, scale: float = 0.5, interp: str = "bicubic"
) -> Image.Image:
    w = max(8, int(round(im.width * scale)))
    h = max(8, int(round(im.height * scale)))
    return im.resize((w, h), _RESAMPLE[interp])


def crop(im: Image.Image, rng: np.random.Generator, frac: float = 0.75) -> Image.Image:
    """Centre crop keeping `frac` of each side."""
    frac = float(min(max(frac, 0.05), 1.0))
    w = max(8, int(round(im.width * frac)))
    h = max(8, int(round(im.height * frac)))
    left = (im.width - w) // 2
    top = (im.height - h) // 2
    return im.crop((left, top, left + w, top + h))


def noise(im: Image.Image, rng: np.random.Generator, sigma: float = 5.0) -> Image.Image:
    """Additive Gaussian noise; `sigma` is on the 0-255 scale."""
    arr = np.asarray(im, dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), "RGB")


def blur(im: Image.Image, rng: np.random.Generator, sigma: float = 1.0) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


OPS: dict[str, Callable[..., Image.Image]] = {
    "jpeg": jpeg,
    "webp": webp,
    "resize": resize,
    "crop": crop,
    "noise": noise,
    "blur": blur,
}


def parse_spec(spec: str) -> list[tuple[str, dict]]:
    """``"resize:scale=0.5+jpeg:q=75"`` -> ``[("resize", {"scale": 0.5}), ("jpeg", {"q": 75.0})]``."""
    spec = spec.strip()
    if spec in ("", CLEAN):
        return []
    steps = []
    for part in spec.split("+"):
        name, _, arg_str = part.strip().partition(":")
        if name not in OPS:
            raise ValueError(f"unknown perturbation '{name}' in spec '{spec}'")
        kwargs: dict = {}
        for item in filter(None, arg_str.split(",")):
            key, _, value = item.partition("=")
            try:
                kwargs[key.strip()] = float(value)
            except ValueError:
                kwargs[key.strip()] = value.strip()
        steps.append((name, kwargs))
    return steps


def ends_with_jpeg(spec: str) -> bool:
    steps = parse_spec(spec)
    return bool(steps) and steps[-1][0] == "jpeg"


def apply(spec: str, im: Image.Image, rng: np.random.Generator | None = None) -> Image.Image:
    rng = rng if rng is not None else np.random.default_rng(0)
    for name, kwargs in parse_spec(spec):
        im = OPS[name](im, rng, **kwargs)
    return im


def rng_for(key: str, spec: str, seed: int = 0) -> np.random.Generator:
    """Reproducible per-image generator, independent of batch order or threads."""
    return np.random.default_rng(zlib.crc32(f"{seed}|{spec}|{key}".encode()))


# Default robustness grid. "mild" conditions are what ordinary sharing does to an
# image and are gated by acceptance thresholds; "severe" ones are reported only.
DEFAULT_GRID: list[dict] = [
    {"family": "jpeg", "spec": "jpeg:q=95", "severity": "mild"},
    {"family": "jpeg", "spec": "jpeg:q=85", "severity": "mild"},
    {"family": "jpeg", "spec": "jpeg:q=75", "severity": "mild"},
    {"family": "jpeg", "spec": "jpeg:q=65", "severity": "severe"},
    {"family": "jpeg", "spec": "jpeg:q=50", "severity": "severe"},
    {"family": "resize", "spec": "resize:scale=0.75", "severity": "mild"},
    {"family": "resize", "spec": "resize:scale=0.5", "severity": "mild"},
    {"family": "resize", "spec": "resize:scale=1.5", "severity": "mild"},
    {"family": "resize", "spec": "resize:scale=0.25", "severity": "severe"},
    {"family": "crop", "spec": "crop:frac=0.9", "severity": "mild"},
    {"family": "crop", "spec": "crop:frac=0.75", "severity": "mild"},
    {"family": "crop", "spec": "crop:frac=0.5", "severity": "severe"},
    {"family": "noise", "spec": "noise:sigma=2", "severity": "mild"},
    {"family": "noise", "spec": "noise:sigma=5", "severity": "mild"},
    {"family": "noise", "spec": "noise:sigma=10", "severity": "severe"},
    {"family": "blur", "spec": "blur:sigma=0.5", "severity": "mild"},
    {"family": "blur", "spec": "blur:sigma=1.0", "severity": "mild"},
    {"family": "blur", "spec": "blur:sigma=2.0", "severity": "severe"},
    {"family": "combined", "spec": "resize:scale=0.5+jpeg:q=75", "severity": "mild"},
]


# ---------------------------------------------------------------------------
# Training-time augmentation
# ---------------------------------------------------------------------------

DEFAULT_AUGMENT = {
    "p_resize": 0.3,
    "resize_scale": [0.4, 1.6],
    "p_blur": 0.3,
    "blur_sigma": [0.1, 2.5],
    "p_noise": 0.2,
    "noise_sigma": [1.0, 8.0],
    "p_jpeg": 0.5,
    "jpeg_quality": [40, 100],
}


def random_degrade(
    im: Image.Image, rng: np.random.Generator, cfg: dict | None = None
) -> Image.Image:
    """Randomly resize, blur, add noise and re-compress, in that order.

    Applied identically to real and generated images so the detector cannot use
    "has been post-processed" as evidence for either class. JPEG comes last
    because that is the order in a real re-share pipeline.
    """
    c = {**DEFAULT_AUGMENT, **(cfg or {})}
    if rng.random() < c["p_resize"]:
        interp = str(rng.choice(["bilinear", "bicubic", "lanczos"]))
        im = resize(im, rng, scale=float(rng.uniform(*c["resize_scale"])), interp=interp)
    if rng.random() < c["p_blur"]:
        im = blur(im, rng, sigma=float(rng.uniform(*c["blur_sigma"])))
    if rng.random() < c["p_noise"]:
        im = noise(im, rng, sigma=float(rng.uniform(*c["noise_sigma"])))
    if rng.random() < c["p_jpeg"]:
        lo, hi = c["jpeg_quality"]
        im = jpeg(im, rng, q=int(rng.integers(int(lo), int(hi) + 1)))
    return im
