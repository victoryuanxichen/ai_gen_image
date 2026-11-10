"""End-to-end tests on synthetic data with the CPU baseline: manifest -> train ->
evaluate -> report -> inference. They check the plumbing, not detection quality."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aidetect.evaluate import run_evaluation
from aidetect.inference import ImageDetector
from aidetect.manifest import (
    ManifestError,
    audit_manifest,
    balanced_weights,
    build_manifest,
    load_registry,
    read_manifest,
)
from aidetect.scoring import load_bundle, prepare_image, score_paths

SMALL_EVAL = {
    "workers": 2,
    "bootstrap": 20,
    "bias": {"min_per_label": 4},
    "robustness": {
        "max_per_label": 12,
        "grid": [
            {"family": "jpeg", "spec": "jpeg:q=75", "severity": "mild"},
            {"family": "noise", "spec": "noise:sigma=10", "severity": "severe"},
        ],
    },
}


# ---- manifest ---------------------------------------------------------------

def test_manifest_splits_and_roles(toy):
    df, audit = toy["manifest"], toy["audit"]
    assert audit["errors"] == []
    assert set(df["split"]) == {"train", "val", "test"}
    fit = df[df["split"].isin(["train", "val"])]
    assert set(fit["generator"]) == {"gen_a"} and set(fit["role"]) == {"seen"}
    test = df[df["split"] == "test"]
    assert set(test["role"]) == {"seen", "unseen"}
    assert set(test.loc[test["role"] == "unseen", "generator"]) == {"gen_b"}
    assert not df["path"].duplicated().any()
    assert set(df["category_group"]) == {"000", "001", "002", "003"}
    assert set(df.loc[df["label"] == 0, "fmt"]) == {"JPEG"}
    assert set(df.loc[df["label"] == 1, "fmt"]) == {"PNG"}
    assert any("JPEG" in w for w in audit["warnings"])  # the format shortcut is flagged


def test_manifest_is_deterministic(toy):
    again, _ = build_manifest(load_registry(toy["registry"]))
    pd.testing.assert_frame_equal(again, toy["manifest"])


def test_manifest_roundtrips_through_csv(toy, tmp_path):
    path = tmp_path / "m.csv"
    toy["manifest"].to_csv(path, index=False)
    df = read_manifest(path)
    assert list(df["category_group"]) == list(toy["manifest"]["category_group"])


def test_unseen_source_cannot_be_used_for_training(toy):
    reg = copy.deepcopy(load_registry(toy["registry"]))
    reg["sources"][2]["split"] = "train"
    with pytest.raises(ManifestError, match="unseen generators must never reach training"):
        build_manifest(reg)


def test_missing_folder_is_an_error_unless_optional(toy):
    reg = copy.deepcopy(load_registry(toy["registry"]))
    reg["sources"].append({**reg["sources"][2], "name": "gone", "path": "nope"})
    with pytest.raises(ManifestError, match="folder not found"):
        build_manifest(reg)
    reg["sources"][-1]["optional"] = True
    _, audit = build_manifest(reg)
    assert any("skipped" in w for w in audit["warnings"])


def test_leakage_between_train_and_test_is_detected(toy, tmp_path):
    root = tmp_path / "leaky"
    shutil.copytree(toy["root"], root)
    victim = sorted((root / "gen_a/train/nature").iterdir())[0]
    shutil.copy(victim, root / "gen_a/val/nature" / victim.name)
    reg = yaml.safe_load((root / "registry.yaml").read_text())
    reg["root"] = str(root)
    _, audit = build_manifest(reg)
    assert any("byte-identical" in e for e in audit["errors"])


def test_audit_flags_generator_in_both_training_and_unseen(toy):
    df = toy["manifest"].copy()
    df.loc[df["generator"] == "gen_b", "generator"] = "gen_a"
    assert any("also appear in training" in e for e in audit_manifest(df)["errors"])


def test_max_per_label_and_real_shard(toy):
    reg = copy.deepcopy(load_registry(toy["registry"]))
    reg["sources"][0]["max_per_label"] = 10
    df, _ = build_manifest(reg)
    part = df[df["source"] == "toy_gen_a_train"]
    assert (part["label"] == 0).sum() == 10 and (part["label"] == 1).sum() == 10

    reg = copy.deepcopy(load_registry(toy["registry"]))
    shards = []
    for k in range(2):
        r = copy.deepcopy(reg)
        r["sources"] = [{**r["sources"][2], "real_shard": [k, 2]}]
        df, _ = build_manifest(r)
        shards.append(set(df.loc[df["label"] == 0, "path"]))
    assert shards[0] and shards[1] and not (shards[0] & shards[1])
    assert len(shards[0] | shards[1]) == 32


def test_balanced_weights():
    df = pd.DataFrame({
        "label": [0] * 10 + [1] * 30 + [1] * 10,
        "source": ["r"] * 10 + ["a"] * 30 + ["b"] * 10,
        "generator": ["r"] * 10 + ["a"] * 30 + ["b"] * 10,
    })
    w = balanced_weights(df)
    assert w.sum() == pytest.approx(1.0)
    assert w[:10].sum() == pytest.approx(0.5)       # real : generated = 1 : 1
    assert w[10:40].sum() == pytest.approx(0.25)    # generators share equally
    assert w[40:].sum() == pytest.approx(0.25)


# ---- scoring ----------------------------------------------------------------

def test_prepare_image_harmonizes_png_but_not_jpeg(toy):
    df = toy["manifest"]
    png = df.loc[df["fmt"] == "PNG", "path"].iloc[0]
    jpg = df.loc[df["fmt"] == "JPEG", "path"].iloc[0]
    from aidetect.imageio import open_image

    assert np.array_equal(np.asarray(prepare_image(jpg, 96)), np.asarray(open_image(jpg)[0]))
    assert not np.array_equal(np.asarray(prepare_image(png, 96)), np.asarray(open_image(png)[0]))
    assert np.array_equal(np.asarray(prepare_image(png, None)), np.asarray(open_image(png)[0]))


def test_score_paths_marks_unreadable_files_nan(toy, baseline_bundle, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    good = toy["manifest"]["path"].iloc[0]
    scores = score_paths([good, str(bad), good], load_bundle(baseline_bundle)["scorer"], workers=2)
    assert np.isnan(scores[1]) and not np.isnan(scores[0]) and scores[0] == scores[2]


# ---- evaluation and report --------------------------------------------------

@pytest.fixture(scope="module")
def evaluation(toy, baseline_bundle, tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    results = run_evaluation(toy["manifest"], load_bundle(baseline_bundle), SMALL_EVAL, out)
    return results, out


def test_evaluation_covers_all_four_acceptance_items(evaluation):
    results, out = evaluation
    assert results["n_test_images"] == 128 and results["n_unreadable"] == 0
    h = results["holdout"]
    assert (h["n_real"], h["n_fake"]) == (32, 32)
    for key in ("precision", "recall", "f1", "auc"):
        assert 0.0 <= h[key] <= 1.0 and key in h["ci95"]
    unseen = results["unseen"]["unseen"]
    assert [r["generator"] for r in unseen["per_generator"]] == ["gen_b"]
    assert unseen["auc_drop_vs_holdout"] == pytest.approx(h["auc"] - unseen["macro_auc"])
    assert results["unseen"]["near_seen"]["per_generator"] == []
    for scope in ("holdout", "unseen"):
        r = results["robustness"][scope]
        assert r["n"] == 24 and [row["spec"] for row in r["rows"]] == ["jpeg:q=75", "noise:sigma=10"]
        for row in r["rows"]:
            assert 0.0 <= row["flip_rate"] <= 1.0
    cat = results["bias"]["holdout"]["by_category_group"]
    assert [row["group"] for row in cat["rows"]] == ["000", "001", "002", "003"]
    assert cat["coverage"] == 1.0
    assert set(results["bias"]) == {"holdout", "all_test"}
    assert results["verdict"] in {"PASS", "FAIL", "INCOMPLETE"}
    names = " ".join(c["check"] for c in results["acceptance"])
    for word in ("hold-out", "unseen", "mild", "category"):
        assert word in names


def test_the_baseline_learns_the_seen_toy_generator(evaluation):
    # The toy artifact is easy by construction; failing here means the pipeline
    # is feeding the model wrong labels or wrong images.
    assert evaluation[0]["holdout"]["auc"] > 0.95


def test_report_and_results_files(evaluation):
    results, out = evaluation
    report = (out / "REPORT.md").read_text()
    for heading in (
        "## Acceptance", "## 1. Hold-out test set", "## 2. Generators not used in training",
        "## 3. Robustness to image processing", "## 4. Bias analysis",
    ):
        assert heading in report
    n_cols = None  # every row of a table has the same cell count as its header
    for line in report.splitlines():
        if not line.startswith("|"):
            n_cols = None
        elif n_cols is None:
            n_cols = line.count("|")
        else:
            assert line.count("|") == n_cols, line
    saved = json.loads((out / "results.json").read_text())  # valid JSON, NaN mapped to null
    assert saved["verdict"] == results["verdict"]


def test_evaluation_resumes_from_cached_predictions(toy, baseline_bundle, evaluation):
    results, out = evaluation
    bundle = load_bundle(baseline_bundle)

    class Exploding:
        name = "exploding"

        def score(self, images):
            raise AssertionError("scores should have come from the cache")

    bundle["scorer"] = Exploding()
    again = run_evaluation(toy["manifest"], bundle, SMALL_EVAL, out)
    assert again["holdout"]["auc"] == results["holdout"]["auc"]


def test_evaluation_requires_a_test_split(toy, baseline_bundle, tmp_path):
    df = toy["manifest"]
    with pytest.raises(ValueError, match="no rows with split == 'test'"):
        run_evaluation(df[df["split"] != "test"], load_bundle(baseline_bundle), SMALL_EVAL, tmp_path)


# ---- inference --------------------------------------------------------------

def test_inference_module(toy, baseline_bundle):
    detector = ImageDetector.load(str(baseline_bundle))
    df = toy["manifest"]
    test = df[(df["split"] == "test") & (df["role"] == "seen")]
    for label, name in ((0, "Real"), (1, "AI-Generated")):
        path = test.loc[test["label"] == label, "path"].iloc[0]
        pred = detector.predict(path)
        assert pred.label == name
        assert 0.0 <= pred.ai_probability <= 1.0 and 0.5 <= pred.confidence <= 1.0
        assert detector.predict(Path(path).read_bytes()).ai_probability == pytest.approx(pred.ai_probability)


def test_confidence_follows_the_threshold(baseline_bundle):
    detector = ImageDetector.load(str(baseline_bundle))
    detector.threshold = 0.3
    ai = detector._prediction(0.4)
    assert ai.label == "AI-Generated" and ai.confidence == pytest.approx(0.4)
    real = detector._prediction(0.2)
    assert real.label == "Real" and real.confidence == pytest.approx(0.8)


def test_cli_end_to_end(toy, tmp_path, capsys):
    from aidetect.cli import main

    manifest = tmp_path / "manifest.csv"
    assert main(["manifest", "--registry", str(toy["registry"]), "--out", str(manifest)]) == 0
    assert (tmp_path / "manifest.audit.json").exists()
    assert main(["train-baseline", "--manifest", str(manifest), "--out", str(tmp_path / "run"), "--workers", "2"]) == 0
    model = tmp_path / "run" / "baseline.joblib"
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(yaml.safe_dump(SMALL_EVAL))
    code = main(["evaluate", "--manifest", str(manifest), "--model", str(model), "--config", str(cfg),
                 "--out", str(tmp_path / "eval"), "--quiet", "--no-fail"])
    assert code == 0 and (tmp_path / "eval" / "REPORT.md").exists()
    capsys.readouterr()
    image = toy["manifest"]["path"].iloc[0]
    assert main(["predict", "--model", str(model), image]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["label"] in {"Real", "AI-Generated"} and 0 <= line["confidence"] <= 1
