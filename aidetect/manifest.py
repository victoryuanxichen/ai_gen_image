"""Build the dataset manifest: one CSV row per image, with everything the rest
of the pipeline needs (label, generator, role, split, category, format, size).

The manifest is the single source of truth for which image is used where. It is
built from a registry YAML that lists dataset folders; see configs/ and
docs/DATASET.md.

Roles
  seen       generators the detector is trained on
  near_seen  not trained on, but the same model family as a seen generator
             (reported separately so they cannot inflate the generalization score)
  unseen     generators from families or versions never used in training

Splits
  train / val  only ever contain `seen` sources
  test         hold-out images; `seen` test rows are the in-distribution test set
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .categories import (
    UNKNOWN,
    CategoryResolver,
    load_group_file,
    load_synsets,
    load_val_labels,
)
from .imageio import IMAGE_EXTENSIONS, probe_image

ROLES = ("seen", "near_seen", "unseen")
COLUMNS = [
    "path", "label", "source", "dataset", "generator", "family", "role", "split",
    "category", "category_group", "fmt", "width", "height",
]


class ManifestError(RuntimeError):
    pass


def _hash01(text: str, salt: str = "") -> float:
    """Stable pseudo-random number in [0, 1) derived from a string."""
    digest = hashlib.md5(f"{salt}|{text}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _list_images(base: Path, pattern: str) -> list[Path]:
    return sorted(
        p for p in base.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_registry(path: str | Path) -> dict:
    with open(path) as fh:
        reg = yaml.safe_load(fh)
    if not isinstance(reg, dict) or "sources" not in reg:
        raise ManifestError(f"{path}: registry must be a mapping with a 'sources' list")
    return reg


def build_manifest(registry: dict, probe: bool = True, workers: int = 16) -> tuple[pd.DataFrame, dict]:
    """Scan the folders named in `registry` and return ``(manifest, audit)``."""
    root = Path(registry.get("root", "."))
    seed = str(registry.get("seed", 0))
    val_fraction = float(registry.get("val_fraction", 0.05))

    synsets = load_synsets(registry["imagenet_synsets"]) if registry.get("imagenet_synsets") else {}
    val_labels = (
        load_val_labels(registry["imagenet_val_labels"])
        if registry.get("imagenet_val_labels") else {}
    )
    groups = (
        load_group_file(registry["category_groups_file"])
        if registry.get("category_groups_file") else {}
    )
    resolver = CategoryResolver(synsets, val_labels, groups)

    warnings: list[str] = []
    rows: list[dict] = []
    for src in registry["sources"]:
        name = src["name"]
        role = src.get("role", "seen")
        split = src.get("split", "test")
        if role not in ROLES:
            raise ManifestError(f"source '{name}': role must be one of {ROLES}")
        if split not in ("train", "val", "test"):
            raise ManifestError(f"source '{name}': split must be train, val or test")
        if role != "seen" and split != "test":
            raise ManifestError(
                f"source '{name}': role '{role}' may only be used with split 'test' "
                "(unseen generators must never reach training or validation)"
            )
        base = root / src["path"]
        if not base.is_dir():
            if src.get("optional"):
                warnings.append(f"optional source '{name}' skipped: {base} not found")
                continue
            raise ManifestError(f"source '{name}': folder not found: {base}")

        cap = src.get("max_per_label")
        for label, glob_key, rule_key in ((0, "real_glob", "category_real"), (1, "fake_glob", "category_fake")):
            pattern = src.get(glob_key)
            if not pattern:
                continue
            files = _list_images(base, pattern)
            if not files:
                warnings.append(f"source '{name}': no images matched {glob_key}='{pattern}'")
            shard = src.get("real_shard") if label == 0 else None
            if shard:
                # Several generators can share one pool of real images. Each source
                # takes its own disjoint slice so no real image is counted twice.
                k, n = int(shard[0]), int(shard[1])
                files = [p for p in files if int(_hash01(str(p.relative_to(root)), "shard") * n) == k]
            if cap and len(files) > int(cap):
                # Deterministic subsample that does not depend on listing order.
                files = sorted(files, key=lambda p: _hash01(str(p.relative_to(root)), seed))[: int(cap)]
                files.sort()
            rule = src.get(rule_key, src.get("category"))
            for p in files:
                rel = str(p.relative_to(root))
                row_split = split
                if split == "train" and _hash01(rel, seed + "|val") < val_fraction:
                    row_split = "val"
                category, group = resolver.resolve(p, rule)
                rows.append({
                    "path": str(p),
                    "label": label,
                    "source": name,
                    "dataset": src.get("dataset", name),
                    "generator": src.get("generator", name),
                    "family": src.get("family", UNKNOWN),
                    "role": role,
                    "split": row_split,
                    "category": category,
                    "category_group": group,
                })

    if not rows:
        raise ManifestError("no images found; check 'root' and the source paths")
    df = pd.DataFrame(rows)

    # ImageNet wnids can be mapped to class indices without the devkit file when
    # all 1000 are present, because the index is the sorted position of the wnid.
    if resolver.unmapped_wnids:
        wnids = sorted(resolver.unmapped_wnids)
        if len(wnids) == 1000:
            from .categories import imagenet_group

            index = {w: i for i, w in enumerate(wnids)}
            mask = df["category"].isin(index)
            idx = df.loc[mask, "category"].map(index)
            df.loc[mask, "category_group"] = idx.map(imagenet_group).values
            df.loc[mask, "category"] = idx.map(lambda i: f"{i:04d}").values
        else:
            warnings.append(
                f"{len(wnids)} distinct ImageNet wnids found but no 'imagenet_synsets' file given; "
                "their category group is 'unknown'. Set imagenet_synsets to LOC_synset_mapping.txt."
            )

    if probe:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            probed = list(pool.map(_probe_safe, df["path"]))
        df["fmt"] = [x[0] for x in probed]
        df["width"] = [x[1] for x in probed]
        df["height"] = [x[2] for x in probed]
        bad = df["fmt"] == "UNREADABLE"
        if bad.any():
            warnings.append(f"{int(bad.sum())} unreadable files dropped")
            df = df[~bad].reset_index(drop=True)
    else:
        df["fmt"] = df["path"].map(lambda p: _fmt_from_suffix(Path(p).suffix))
        df["width"] = 0
        df["height"] = 0

    df = df[COLUMNS].sort_values(["split", "source", "label", "path"]).reset_index(drop=True)
    audit = audit_manifest(df)
    audit["warnings"] = warnings + audit["warnings"]
    return df, audit


def _probe_safe(path: str) -> tuple[str, int, int]:
    try:
        return probe_image(path)
    except Exception:
        return "UNREADABLE", 0, 0


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return f"unreadable:{path}"
    return h.hexdigest()


def _fmt_from_suffix(suffix: str) -> str:
    return {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}.get(
        suffix.lower(), suffix.lstrip(".").upper()
    )


def audit_manifest(df: pd.DataFrame) -> dict:
    """Checks that make data problems visible before any GPU time is spent."""
    errors: list[str] = []
    warnings: list[str] = []

    dup = df["path"].duplicated()
    if dup.any():
        errors.append(f"{int(dup.sum())} image paths are listed more than once")

    # An unseen generator must not appear in training in any source.
    fit = df[df["split"].isin(["train", "val"])]
    test = df[df["split"] == "test"]
    trained_generators = set(fit.loc[fit["label"] == 1, "generator"])
    for role in ("near_seen", "unseen"):
        clash = trained_generators & set(test.loc[test["role"] == role, "generator"])
        if clash:
            errors.append(f"generators marked {role} also appear in training data: {sorted(clash)}")

    # The same file name on both sides of the split usually means the same image
    # (ImageNet files keep their names in every benchmark built on them).
    # Many benchmarks also reuse short numeric names, so a name collision is only
    # a candidate: it counts as leakage when the file contents are identical.
    fit_by_name: dict[str, list[str]] = {}
    for p in fit["path"]:
        fit_by_name.setdefault(Path(p).name, []).append(p)
    candidates = [p for p in test["path"] if Path(p).name in fit_by_name]
    max_checked = 20000
    if len(candidates) > max_checked:
        warnings.append(
            f"{len(candidates)} test files share a name with a training file; "
            f"contents compared for the first {max_checked} only"
        )
        candidates = candidates[:max_checked]
    leaked = []
    fit_hash_cache: dict[str, str] = {}
    for p in candidates:
        h = _file_hash(p)
        for q in fit_by_name[Path(p).name]:
            if q not in fit_hash_cache:
                fit_hash_cache[q] = _file_hash(q)
            if fit_hash_cache[q] == h:
                leaked.append(p)
                break
    if leaked:
        errors.append(
            f"{len(leaked)} test images are byte-identical to a train/val image (leakage), "
            f"e.g. {leaked[:3]}"
        )

    counts = (
        df.groupby(["split", "role", "generator", "label"]).size().rename("n").reset_index()
    )

    # File-format and size shortcuts between the two classes.
    fmt_share = {}
    for split, part in df.groupby("split"):
        share = {}
        for label, sub in part.groupby("label"):
            share[int(label)] = float((sub["fmt"] == "JPEG").mean())
        fmt_share[split] = share
        if 0 in share and 1 in share and abs(share[0] - share[1]) > 0.2:
            warnings.append(
                f"{split}: {share[0]:.0%} of real images are JPEG vs {share[1]:.0%} of generated "
                "images. Keep harmonize_jpeg_quality enabled or the detector can learn the file format."
            )
    size = {}
    if (df["width"] > 0).any():
        short = df[["width", "height"]].min(axis=1)
        med = short.groupby([df["generator"], df["label"]]).median()
        size = {f"{g}|{int(l)}": float(v) for (g, l), v in med.items()}
        for gen in df["generator"].unique():
            a, b = size.get(f"{gen}|0"), size.get(f"{gen}|1")
            if a and b and max(a, b) / min(a, b) > 1.5:
                warnings.append(
                    f"{gen}: median short side is {a:.0f}px for real vs {b:.0f}px for generated images "
                    "(size shortcut; random-resize augmentation and common-size preprocessing mitigate it)"
                )

    coverage = (
        (df["category_group"] != UNKNOWN).groupby([df["source"], df["label"]]).mean()
    )
    category_coverage = {f"{s}|{int(l)}": float(v) for (s, l), v in coverage.items()}
    for key, value in category_coverage.items():
        if value < 0.5:
            warnings.append(
                f"category known for only {value:.0%} of {key.replace('|0', ' real').replace('|1', ' generated')} "
                "images; fix the category rule or the bias analysis will skip them"
            )

    return {
        "n_images": int(len(df)),
        "errors": errors,
        "warnings": warnings,
        "counts": counts.to_dict(orient="records"),
        "jpeg_share_by_split_and_label": fmt_share,
        "median_short_side_by_generator_and_label": size,
        "category_coverage_by_source_and_label": category_coverage,
    }


def balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """Sampling weights giving real:generated = 1:1, with generated images spread
    evenly over generators and real images spread evenly over sources.

    Without this, the generator with the most files dominates every batch.
    """
    weights = np.zeros(len(df), dtype=np.float64)
    for label, key in ((0, "source"), (1, "generator")):
        mask = (df["label"] == label).to_numpy()
        if not mask.any():
            continue
        groups = df.loc[mask, key]
        per_group = groups.map(groups.value_counts())
        weights[mask] = 0.5 / (groups.nunique() * per_group.to_numpy())
    total = weights.sum()
    return weights / total if total > 0 else weights


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"category": str, "category_group": str})
    missing = [c for c in ("path", "label", "generator", "role", "split") if c not in df.columns]
    if missing:
        raise ManifestError(f"{path}: manifest is missing columns {missing}")
    for col in ("category", "category_group"):
        if col not in df.columns:
            df[col] = UNKNOWN
        df[col] = df[col].fillna(UNKNOWN)
    return df
