"""Tests for the parts that need no GPU: image I/O, perturbations, metrics, calibration."""

import io
import math

import numpy as np
import pytest
from PIL import Image

from aidetect import perturb
from aidetect.calibrate import Calibrator, threshold_for_fpr
from aidetect.categories import CategoryResolver, imagenet_group
from aidetect.imageio import ImageError, harmonize, open_image
from aidetect.metrics import (
    binary_metrics,
    bootstrap_ci,
    expected_calibration_error,
    stability,
    tpr_at_fpr,
)


def _image(w=96, h=80, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def _bytes(im, fmt, **kw):
    buf = io.BytesIO()
    im.save(buf, format=fmt, **kw)
    return buf.getvalue()


# ---- image I/O --------------------------------------------------------------

def test_open_image_reports_format_and_rgb():
    im, fmt = open_image(_bytes(_image(), "PNG"))
    assert (fmt, im.mode, im.size) == ("PNG", "RGB", (96, 80))
    _, fmt = open_image(_bytes(_image(), "JPEG"))
    assert fmt == "JPEG"


def test_open_image_flattens_transparency_onto_white():
    rgba = Image.new("RGBA", (16, 16), (255, 0, 0, 0))  # fully transparent red
    im, _ = open_image(_bytes(rgba, "PNG"))
    assert im.mode == "RGB" and im.getpixel((0, 0)) == (255, 255, 255)


def test_open_image_handles_grayscale_and_palette():
    assert open_image(_bytes(_image().convert("L"), "PNG"))[0].mode == "RGB"
    assert open_image(_bytes(_image().convert("P"), "PNG"))[0].mode == "RGB"


def test_open_image_applies_exif_orientation():
    im = _image(96, 80)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    out, _ = open_image(_bytes(im, "JPEG", exif=exif))
    assert out.size == (80, 96)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n broken"])
def test_open_image_rejects_garbage(data):
    with pytest.raises(ImageError):
        open_image(data)


def test_open_image_rejects_tiny_images():
    with pytest.raises(ImageError):
        open_image(_bytes(_image(4, 4), "PNG"))


def test_open_image_rejects_oversized_images(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(ImageError):
        open_image(_bytes(_image(64, 64), "PNG"))    # 4096 px: above the limit
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)
    assert open_image(_bytes(_image(64, 64), "PNG"))[0].size == (64, 64)


def test_harmonize_only_touches_non_jpeg():
    im = _image()
    assert harmonize(im, "JPEG", 96) is im
    assert harmonize(im, "PNG", None) is im
    out = harmonize(im, "PNG", 96)
    assert out is not im and out.size == im.size and out.mode == "RGB"


# ---- perturbations ----------------------------------------------------------

def test_parse_spec():
    assert perturb.parse_spec("clean") == []
    assert perturb.parse_spec("resize:scale=0.5+jpeg:q=75") == [
        ("resize", {"scale": 0.5}), ("jpeg", {"q": 75.0}),
    ]
    assert perturb.ends_with_jpeg("resize:scale=0.5+jpeg:q=75")
    assert not perturb.ends_with_jpeg("jpeg:q=75+blur:sigma=1")
    with pytest.raises(ValueError):
        perturb.parse_spec("sharpen:amount=2")


def test_every_grid_entry_runs_and_keeps_rgb():
    im = _image(120, 90)
    for entry in perturb.DEFAULT_GRID:
        out = perturb.apply(entry["spec"], im, np.random.default_rng(0))
        assert out.mode == "RGB" and min(out.size) >= 8, entry["spec"]


def test_geometry():
    im = _image(100, 60)
    assert perturb.apply("resize:scale=0.5", im).size == (50, 30)
    assert perturb.apply("crop:frac=0.5", im).size == (50, 30)
    assert perturb.apply("clean", im) is im


def test_noise_is_reproducible_per_image_and_spec():
    im = _image()
    a = perturb.apply("noise:sigma=5", im, perturb.rng_for("a.png", "noise:sigma=5", 0))
    b = perturb.apply("noise:sigma=5", im, perturb.rng_for("a.png", "noise:sigma=5", 0))
    c = perturb.apply("noise:sigma=5", im, perturb.rng_for("b.png", "noise:sigma=5", 0))
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_noise_strength_matches_sigma():
    flat = Image.new("RGB", (128, 128), (128, 128, 128))
    out = np.asarray(perturb.apply("noise:sigma=10", flat, np.random.default_rng(0)), dtype=float)
    assert 9.0 < out.std() < 11.0


def test_random_degrade_returns_rgb_images():
    rng = np.random.default_rng(0)
    for _ in range(30):
        out = perturb.random_degrade(_image(), rng)
        assert out.mode == "RGB" and min(out.size) >= 8


# ---- metrics ----------------------------------------------------------------

def test_binary_metrics_known_values():
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1]
    m = binary_metrics(y, p, 0.5)  # tp=3 fn=1 fp=1 tn=3
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(0.25)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert m["auc"] == pytest.approx(13.5 / 16)
    assert (m["n"], m["n_real"], m["n_fake"]) == (8, 4, 4)


def test_binary_metrics_single_class_and_nan_scores():
    m = binary_metrics([1, 1, 1], [0.9, 0.2, float("nan")], 0.5)
    assert m["n"] == 2 and m["recall"] == pytest.approx(0.5)
    assert math.isnan(m["auc"]) and math.isnan(m["fpr"])
    m = binary_metrics([0, 0], [0.1, 0.2], 0.5)
    assert m["fpr"] == 0.0 and math.isnan(m["recall"]) and math.isnan(m["f1"])


def test_threshold_is_inclusive():
    assert binary_metrics([1], [0.5], 0.5)["recall"] == 1.0


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 300)
    p = np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1)
    m = binary_metrics(y, p, 0.5)
    ci = bootstrap_ci(y, p, 0.5, n_boot=100)
    for key in ("precision", "recall", "f1", "auc"):
        lo, hi = ci[key]
        assert lo <= m[key] <= hi and hi - lo < 0.2


def test_ece_perfect_and_bad():
    y = np.array([0] * 50 + [1] * 50)
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)
    assert expected_calibration_error(y, 1.0 - y) == pytest.approx(1.0)


def test_tpr_at_fpr():
    y = np.array([0] * 100 + [1] * 100)
    p = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.4, 1.0, 100)])
    assert 0.8 < tpr_at_fpr(y, p, 0.01) <= 1.0
    assert math.isnan(tpr_at_fpr([1, 1], [0.2, 0.3]))


def test_stability():
    s = stability([0.9, 0.1, 0.6, 0.4], [0.8, 0.2, 0.4, 0.4], 0.5)
    assert s["flip_rate"] == pytest.approx(0.25)
    assert s["mean_abs_prob_change"] == pytest.approx(0.1)


# ---- calibration ------------------------------------------------------------

def test_calibrator_fit_improves_calibration_and_keeps_order():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    true_logit = rng.normal(2.0 * (2 * y - 1), 1.5)
    overconfident = 5.0 * true_logit + 3.0
    cal = Calibrator.fit(overconfident, y)
    assert cal.a > 0
    before = expected_calibration_error(y, Calibrator()(overconfident))
    after = expected_calibration_error(y, cal(overconfident))
    assert after < before and after < 0.03
    assert np.all(np.diff(cal(np.sort(overconfident))) >= 0)


def test_calibrator_falls_back_to_identity():
    assert Calibrator.fit([1.0, 2.0], [1, 1]) == Calibrator()
    assert Calibrator.from_dict(None) == Calibrator()
    assert Calibrator.from_dict(Calibrator(2.0, -1.0).to_dict()) == Calibrator(2.0, -1.0)


def test_threshold_for_fpr_respects_the_target():
    rng = np.random.default_rng(0)
    p_real = rng.uniform(0, 1, 1000)
    for target in (0.01, 0.05, 0.2):
        thr = threshold_for_fpr(p_real, target)
        assert (p_real >= thr).mean() <= target


# ---- categories -------------------------------------------------------------

def test_imagenet_groups():
    assert imagenet_group(0) == "animal_other"
    assert imagenet_group(207) == "dog"          # golden retriever
    assert imagenet_group(417) == "object"       # balloon
    assert imagenet_group(963) == "food"         # pizza
    assert imagenet_group(980) == "landscape"    # volcano
    assert imagenet_group(985) == "plant_fungus" # daisy
    assert imagenet_group(5000) == "unknown"


def test_category_resolver(tmp_path):
    from pathlib import Path

    r = CategoryResolver(synsets={"n02099601": 207}, val_labels={"ILSVRC2012_val_00000001": "n02099601"})
    index_rule = {"from": "regex", "pattern": r"^(\d+)_", "kind": "imagenet_index"}
    assert r.resolve(Path("x/207_sdv4_00001.png"), index_rule) == ("0207", "dog")
    assert r.resolve(Path("x/nomatch.png"), index_rule) == ("unknown", "unknown")
    wnid_rule = {"from": "regex", "pattern": r"^(n\d{8})_", "kind": "wnid"}
    assert r.resolve(Path("x/n02099601_12.JPEG"), wnid_rule) == ("0207", "dog")
    assert r.resolve(Path("x/n00000000_12.JPEG"), wnid_rule) == ("n00000000", "unknown")
    assert r.unmapped_wnids == {"n00000000"}
    assert r.resolve(Path("x/ILSVRC2012_val_00000001.JPEG"), {"from": "val_labels"}) == ("0207", "dog")
    assert r.resolve(Path("root/horse/0_real/a.png"), {"from": "parent", "level": 2}) == ("horse", "horse")
    assert r.resolve(Path("a.png"), None) == ("unknown", "unknown")
