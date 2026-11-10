import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """A tiny GenImage-shaped dataset and its manifest."""
    from make_toy_data import make

    from aidetect.manifest import build_manifest, load_registry

    root = tmp_path_factory.mktemp("toy")
    registry_path = make(root, n_train=64, n_test=32)
    manifest, audit = build_manifest(load_registry(registry_path))
    return {"root": root, "registry": registry_path, "manifest": manifest, "audit": audit}


@pytest.fixture(scope="session")
def baseline_bundle(toy, tmp_path_factory):
    from aidetect.baseline import train_baseline

    out = tmp_path_factory.mktemp("run") / "baseline.joblib"
    return train_baseline(toy["manifest"], out, workers=2)
