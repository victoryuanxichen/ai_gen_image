"""Tests for the deep model, training loop and HTTP service.

They are skipped automatically when PyTorch/timm or FastAPI are not installed.
They use a tiny randomly initialised ViT, so no weights are downloaded and they
run on CPU in about a minute.
"""

import io

import numpy as np
import pytest
from PIL import Image

TINY = {"backbone": "vit_tiny_patch16_224", "head": "linear", "hidden_dim": 32,
        "dropout": 0.0, "unfreeze_last_n": 0, "img_size": 64}


def _image(seed=0, size=(90, 70)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8), "RGB")


def test_detector_freezes_backbone_and_scores_images():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from aidetect.torch_model import Detector, TorchScorer

    model = Detector(**TINY, pretrained=False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("head.") for n in trainable)
    model.train()
    assert not model.backbone.training  # a frozen backbone stays in eval mode

    scorer = TorchScorer(model, model.data_config() | {"image_size": 64}, "cpu")
    scores = scorer.score([_image(0), _image(1, (200, 50))])
    assert scores.shape == (2,) and np.isfinite(scores).all()
    assert model.training  # scoring restores the previous mode

    logits = model(torch.randn(2, 3, 64, 64))
    logits.sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())


def test_unfreeze_last_n():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from aidetect.torch_model import Detector

    model = Detector(**{**TINY, "unfreeze_last_n": 1}, pretrained=False)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("backbone.blocks.11.") for n in names)
    assert not any(n.startswith("backbone.blocks.0.") for n in names)


@pytest.mark.parametrize("save_backbone", [True])
def test_train_save_load_and_predict(toy, tmp_path, save_backbone):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from aidetect.inference import ImageDetector
    from aidetect.train import train

    manifest = tmp_path / "manifest.csv"
    toy["manifest"].to_csv(manifest, index=False)
    cfg = {
        "pretrained": False,
        "model": TINY,
        "data": {"workers": 0, "image_size": 64},
        "train": {"epochs": 1, "batch_size": 16, "amp": False, "log_every": 1},
        "validation": {"max_per_label": 8},
        "save_backbone": save_backbone,
    }
    bundle = train(cfg, str(manifest), str(tmp_path / "run"), device="cpu")
    assert bundle.exists() and (tmp_path / "run" / "history.json").exists()

    detector = ImageDetector.load(str(bundle), device="cpu")
    path = toy["manifest"]["path"].iloc[0]
    first = detector.predict(path)
    assert first.label in {"Real", "AI-Generated"} and 0.0 <= first.ai_probability <= 1.0
    again = ImageDetector.load(str(bundle), device="cpu").predict(path)
    assert again.ai_probability == pytest.approx(first.ai_probability, abs=1e-5)


def test_http_service(baseline_bundle):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from aidetect.inference import ImageDetector
    from aidetect.serve import create_app

    def encode(im, fmt):
        buf = io.BytesIO()
        im.save(buf, format=fmt)
        return buf.getvalue()

    with TestClient(create_app(ImageDetector.load(str(baseline_bundle)))) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        for fmt, mime in (("JPEG", "image/jpeg"), ("PNG", "image/png")):
            r = client.post("/v1/detect", files={"file": (f"x.{fmt.lower()}", encode(_image(), fmt), mime)})
            assert r.status_code == 200
            body = r.json()
            assert body["label"] in {"Real", "AI-Generated"}
            assert 0.0 <= body["confidence"] <= 1.0 and 0.0 <= body["ai_probability"] <= 1.0
        r = client.post("/v1/detect", files={"file": ("x.txt", b"hello", "text/plain")})
        assert r.status_code == 400
        r = client.post("/v1/detect", files={"file": ("x.webp", encode(_image(), "WEBP"), "image/webp")})
        assert r.status_code == 415
