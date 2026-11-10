"""Create a tiny synthetic dataset laid out like GenImage, for smoke tests.

    python scripts/make_toy_data.py toy_data
    python -m aidetect manifest --registry toy_data/registry.yaml --out toy_data/manifest.csv

"Real" images are smooth random textures saved as JPEG at varied sizes. "Generated"
images are similar textures with a faint periodic pattern (a stand-in for
upsampling artifacts) saved as PNG at a fixed size, reproducing the format and
size differences found in the public benchmarks. gen_a is the seen generator,
gen_b (a weaker, different-period artifact) the unseen one.

This data only checks that the pipeline runs. It says nothing about real detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

N_CLASSES = 4  # toy stand-ins for content categories


def texture(rng: np.random.Generator, width: int, height: int, cls: int) -> np.ndarray:
    base = rng.normal(0, 1, (height // 8 + 2, width // 8 + 2, 3)).astype(np.float32)
    im = Image.fromarray(np.clip(128 + 55 * base, 0, 255).astype(np.uint8), "RGB")
    im = im.resize((width, height), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1.0))
    arr = np.asarray(im, dtype=np.float32)
    arr[..., cls % 3] += 25 * (cls + 1) / N_CLASSES  # a class-dependent colour cast
    arr += rng.normal(0, 3.0, arr.shape)             # sensor-like noise
    return arr


def add_grid(arr: np.ndarray, period: int, strength: float) -> np.ndarray:
    h, w, _ = arr.shape
    yy, xx = np.indices((h, w))
    pattern = np.cos(2 * np.pi * xx / period) * np.cos(2 * np.pi * yy / period)
    return arr + strength * pattern[..., None]


def save(arr: np.ndarray, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    if fmt == "JPEG":
        im.save(path, format="JPEG", quality=96)
    else:
        im.save(path, format="PNG")


def make(root: Path, n_train: int = 160, n_test: int = 80, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    generators = {"gen_a": (4, 4.0, 128), "gen_b": (6, 3.0, 192)}  # period, strength, size
    counter = 0
    for gen, (period, strength, size) in generators.items():
        for split, n in (("train", n_train), ("val", n_test)):
            if gen == "gen_b" and split == "train":
                continue  # the unseen generator has no training data
            for i in range(n):
                cls = i % N_CLASSES
                w, h = int(rng.integers(140, 260)), int(rng.integers(140, 260))
                counter += 1
                save(texture(rng, w, h, cls), root / gen / split / "nature" / f"{cls:03d}_real_{counter:06d}.JPEG", "JPEG")
                fake = add_grid(texture(rng, size, size, cls), period, strength)
                save(fake, root / gen / split / "ai" / f"{cls:03d}_{gen}_{i:05d}.png", "PNG")

    registry = {
        "root": str(root.resolve()),
        "seed": 0,
        "val_fraction": 0.25,
        "sources": [
            {"name": "toy_gen_a_train", "dataset": "toy", "path": "gen_a/train", "real_glob": "nature/*",
             "fake_glob": "ai/*", "generator": "gen_a", "family": "toy_gan", "role": "seen", "split": "train",
             "category": {"from": "regex", "pattern": r"^(\d+)_", "kind": "name"}},
            {"name": "toy_gen_a_test", "dataset": "toy", "path": "gen_a/val", "real_glob": "nature/*",
             "fake_glob": "ai/*", "generator": "gen_a", "family": "toy_gan", "role": "seen", "split": "test",
             "category": {"from": "regex", "pattern": r"^(\d+)_", "kind": "name"}},
            {"name": "toy_gen_b_test", "dataset": "toy", "path": "gen_b/val", "real_glob": "nature/*",
             "fake_glob": "ai/*", "generator": "gen_b", "family": "toy_diffusion", "role": "unseen", "split": "test",
             "category": {"from": "regex", "pattern": r"^(\d+)_", "kind": "name"}},
        ],
    }
    path = root / "registry.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False))
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "toy_data")
    print(f"wrote {make(out)}")
