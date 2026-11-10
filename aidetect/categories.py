"""Content-category labels used by the bias analysis.

Each manifest row gets a fine `category` (an ImageNet class index, or a folder
name such as an LSUN class) and a coarse `category_group`. Categories are read
from file names or folder names with rules declared per source in the data
registry, because every public benchmark names its files differently.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

UNKNOWN = "unknown"

# Coarse ImageNet-1k groups by class-index range (index = position of the WordNet
# id in sorted order, the standard ILSVRC-2012 ordering). The ranges follow the
# well-known layout of the index (animals first, dogs at 151-268, then objects,
# food, landscapes, people, plants). They are approximate at the edges: a few
# classes inside the "food" range are tableware, for example. Supply your own
# mapping with `category_groups_file` in the registry for an exact taxonomy.
IMAGENET_GROUP_RANGES: list[tuple[int, int, str]] = [
    (0, 150, "animal_other"),
    (151, 268, "dog"),
    (269, 397, "animal_other"),
    (398, 921, "object"),
    (922, 969, "food"),
    (970, 980, "landscape"),
    (981, 983, "person"),
    (984, 998, "plant_fungus"),
    (999, 999, "object"),
]


def imagenet_group(index: int) -> str:
    for lo, hi, name in IMAGENET_GROUP_RANGES:
        if lo <= index <= hi:
            return name
    return UNKNOWN


def load_synsets(path: str | Path) -> dict[str, int]:
    """Read ImageNet's ``LOC_synset_mapping.txt`` (one ``wnid description`` per
    line, already in index order) into ``{wnid: index}``."""
    wnids = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            wnids.append(line.split()[0])
    return {w: i for i, w in enumerate(wnids)}


def load_val_labels(path: str | Path) -> dict[str, str]:
    """Read a ``file stem -> wnid`` table for ImageNet validation images.

    Accepts Kaggle's ``LOC_val_solution.csv`` (``ImageId,PredictionString`` where
    the prediction string starts with the wnid) or any two-column CSV of
    ``filename,wnid``.
    """
    out: dict[str, str] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[1].strip():
                continue
            wnid = row[1].split()[0]
            if re.fullmatch(r"n\d{8}", wnid):
                out[Path(row[0]).stem] = wnid
    return out


def load_group_file(path: str | Path) -> dict[str, str]:
    """Two-column CSV ``category,group`` overriding the built-in grouping."""
    out: dict[str, str] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0].strip().lower() != "category":
                out[row[0].strip()] = row[1].strip()
    return out


class CategoryResolver:
    """Applies one source's category rule to a file path.

    Rule fields (all optional; no rule means "unknown"):
      from:    "regex" | "parent" | "val_labels" | "none"
      pattern: regex with one capture group, matched against the file name (regex)
      level:   1 = parent folder name, 2 = grandparent, ... (parent)
      kind:    "name" (use the captured text as is), "imagenet_index" or "wnid"
    """

    def __init__(
        self,
        synsets: dict[str, int] | None = None,
        val_labels: dict[str, str] | None = None,
        groups: dict[str, str] | None = None,
    ):
        self.synsets = synsets or {}
        self.val_labels = val_labels or {}
        self.groups = groups or {}
        self.unmapped_wnids: set[str] = set()

    def _finish(self, raw: str | None, kind: str) -> tuple[str, str]:
        if raw is None:
            return UNKNOWN, UNKNOWN
        if kind == "wnid":
            if raw not in self.synsets:
                self.unmapped_wnids.add(raw)
                return raw, self.groups.get(raw, UNKNOWN)
            raw, kind = str(self.synsets[raw]), "imagenet_index"
        if kind == "imagenet_index":
            try:
                idx = int(raw)
            except ValueError:
                return UNKNOWN, UNKNOWN
            cat = f"{idx:04d}"
            return cat, self.groups.get(cat, imagenet_group(idx))
        return raw, self.groups.get(raw, raw)

    def resolve(self, path: Path, rule: dict | None) -> tuple[str, str]:
        """Return ``(category, category_group)`` for one file."""
        if not rule:
            return UNKNOWN, UNKNOWN
        source = rule.get("from", "none")
        kind = rule.get("kind", "name")
        if source == "regex":
            m = re.search(rule["pattern"], path.name)
            return self._finish(m.group(1) if m else None, kind)
        if source == "parent":
            level = int(rule.get("level", 1))
            parts = path.parts
            raw = parts[-1 - level] if len(parts) > level else None
            return self._finish(raw, kind)
        if source == "val_labels":
            return self._finish(self.val_labels.get(path.stem), "wnid")
        return UNKNOWN, UNKNOWN
