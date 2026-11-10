"""The four acceptance evaluations, run on the manifest's `test` split:

1. hold-out      seen generators, images never used in training
2. unseen        generators never used in training, one row per generator
3. robustness    the same images under compression, resizing, cropping, noise, blur
4. bias          results broken down by content category, resolution and file format

All four use the bundle's fixed decision threshold. No threshold is ever tuned
on test data, because the deployed service cannot do that either.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import perturb
from .categories import UNKNOWN
from .manifest import _hash01
from .metrics import (
    binary_metrics,
    bootstrap_ci,
    expected_calibration_error,
    stability,
    tpr_at_fpr,
)
from .scoring import score_paths

DEFAULT_EVAL_CONFIG: dict = {
    "batch_size": 64,
    "workers": 8,
    "seed": 0,
    "bootstrap": 200,
    "robustness": {"max_per_label": 2000, "grid": perturb.DEFAULT_GRID},
    "bias": {"min_per_label": 50},
    # Proposed thresholds; replace with the values agreed for the project.
    "acceptance": {
        "holdout": {"auc": 0.99, "f1": 0.95, "precision": 0.95, "recall": 0.95},
        "unseen": {
            "macro_auc": 0.93,
            "macro_balanced_accuracy": 0.85,
            "worst_auc": 0.80,
            "max_auc_drop": 0.06,
        },
        "robustness_mild": {"max_auc_drop": 0.05, "max_flip_rate": 0.10},
        "fairness": {"max_balanced_accuracy_gap": 0.05, "max_fpr_gap": 0.03},
    },
}


def merge_config(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = value
    return out


def _slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", spec).strip("_") or "clean"


class PredictionCache:
    """Scores per perturbation, stored as CSV so interrupted runs can resume."""

    def __init__(self, bundle: dict, cfg: dict, out_dir: Path, progress: bool = False):
        self.bundle, self.cfg, self.progress = bundle, cfg, progress
        self.dir = out_dir / "predictions"
        self.dir.mkdir(parents=True, exist_ok=True)

    def logits(self, paths: list[str], spec: str = perturb.CLEAN) -> np.ndarray:
        file = self.dir / f"{_slug(spec)}.csv"
        known: dict[str, float] = {}
        if file.exists():
            old = pd.read_csv(file)
            known = dict(zip(old["path"], old["logit"]))
        missing = [p for p in paths if p not in known]
        if missing:
            scores = score_paths(
                missing,
                self.bundle["scorer"],
                harmonize_q=self.bundle["harmonize_q"],
                perturbation=spec,
                batch_size=self.cfg["batch_size"],
                workers=self.cfg["workers"],
                seed=self.cfg["seed"],
                progress=self.progress,
            )
            known.update(zip(missing, scores))
            pd.DataFrame({"path": list(known), "logit": list(known.values())}).to_csv(file, index=False)
        return np.array([known[p] for p in paths], dtype=np.float64)


def _resolution_bucket(width: int, height: int) -> str:
    short = min(width, height)
    if short <= 0:
        return UNKNOWN
    if short < 256:
        return "<256"
    if short < 512:
        return "256-511"
    if short < 1024:
        return "512-1023"
    return ">=1024"


def _sample(df: pd.DataFrame, max_per_label: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, sub in df.groupby("label"):
        if len(sub) > max_per_label:
            order = sub["path"].map(lambda p: _hash01(p, f"robust|{seed}"))
            sub = sub.loc[order.sort_values().index[:max_per_label]]
        parts.append(sub)
    return pd.concat(parts) if parts else df


def _nanmean(values) -> float:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def _nanmin(values) -> float:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.min(values)) if values else float("nan")


def _gap(values) -> float:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.max(values) - np.min(values)) if len(values) >= 2 else float("nan")


def _group_table(df: pd.DataFrame, column: str, thr: float, min_per_label: int) -> dict:
    rows = []
    for value, sub in df.groupby(column):
        if value == UNKNOWN:
            continue
        m = binary_metrics(sub["label"], sub["p"], thr)
        enough_real, enough_fake = m["n_real"] >= min_per_label, m["n_fake"] >= min_per_label
        rows.append({
            "group": str(value),
            "n_real": m["n_real"],
            "n_fake": m["n_fake"],
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"] if enough_real and enough_fake else float("nan"),
            "recall": m["recall"] if enough_fake else float("nan"),
            "fpr": m["fpr"] if enough_real else float("nan"),
            "auc": m["auc"] if enough_real and enough_fake else float("nan"),
        })
    rows.sort(key=lambda r: r["group"])
    return {
        "rows": rows,
        "coverage": float((df[column] != UNKNOWN).mean()) if len(df) else float("nan"),
        "balanced_accuracy_gap": _gap([r["balanced_accuracy"] for r in rows]),
        "recall_gap": _gap([r["recall"] for r in rows]),
        "fpr_gap": _gap([r["fpr"] for r in rows]),
    }


def _check(name: str, value: float, op: str, limit: float) -> dict:
    if value is None or math.isnan(value):
        passed = None
    else:
        passed = value >= limit if op == ">=" else value <= limit
    return {"check": name, "value": value, "op": op, "limit": limit, "passed": passed}


def run_evaluation(
    manifest: pd.DataFrame,
    bundle: dict,
    config: dict | None = None,
    out_dir: str | Path = "eval",
    progress: bool = False,
) -> dict:
    cfg = merge_config(DEFAULT_EVAL_CONFIG, config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = bundle["threshold"]
    calibrate = bundle["calibrator"]
    cache = PredictionCache(bundle, cfg, out_dir, progress)

    test = manifest[manifest["split"] == "test"].copy()
    if test.empty:
        raise ValueError("the manifest has no rows with split == 'test'")
    test["p"] = calibrate(cache.logits(list(test["path"])))
    n_failed = int(test["p"].isna().sum())
    test = test[~test["p"].isna()]
    if "width" in test.columns:
        test["resolution"] = [_resolution_bucket(w, h) for w, h in zip(test["width"], test["height"])]
    else:
        test["resolution"] = UNKNOWN
    if "fmt" not in test.columns:
        test["fmt"] = UNKNOWN

    results: dict = {
        "model": bundle["meta"].get("model_name", getattr(bundle["scorer"], "name", "detector")),
        "threshold": thr,
        "harmonize_jpeg_quality": bundle["harmonize_q"],
        "n_test_images": int(len(test)),
        "n_unreadable": n_failed,
    }

    # 1. Hold-out -----------------------------------------------------------
    seen = test[test["role"] == "seen"]
    holdout = None
    if len(seen):
        holdout = binary_metrics(seen["label"], seen["p"], thr)
        holdout["ece"] = expected_calibration_error(seen["label"], seen["p"])
        holdout["tpr_at_1pct_fpr"] = tpr_at_fpr(seen["label"], seen["p"], 0.01)
        holdout["ci95"] = bootstrap_ci(
            seen["label"], seen["p"], thr, n_boot=int(cfg["bootstrap"]), seed=cfg["seed"]
        )
        holdout["per_generator"] = [
            {"generator": g, **binary_metrics(sub["label"], sub["p"], thr)}
            for g, sub in seen.groupby("generator")
        ]
    results["holdout"] = holdout

    # 2. Unseen generators --------------------------------------------------
    unseen_results = {}
    for role in ("near_seen", "unseen"):
        part = test[test["role"] == role]
        rows = []
        for gen, sub in part.groupby("generator"):
            m = binary_metrics(sub["label"], sub["p"], thr)
            rows.append({"generator": gen, "family": sub["family"].iloc[0], **m})
        rows.sort(key=lambda r: (r["family"], r["generator"]))
        macro_auc = _nanmean([r["auc"] for r in rows])
        macro_bacc = _nanmean([r["balanced_accuracy"] for r in rows])
        unseen_results[role] = {
            "per_generator": rows,
            "macro_auc": macro_auc,
            "macro_balanced_accuracy": macro_bacc,
            "macro_recall": _nanmean([r["recall"] for r in rows]),
            "macro_fpr": _nanmean([r["fpr"] for r in rows]),
            "worst_auc": _nanmin([r["auc"] for r in rows]),
            "auc_drop_vs_holdout": (holdout["auc"] - macro_auc) if holdout else float("nan"),
            "balanced_accuracy_drop_vs_holdout": (
                holdout["balanced_accuracy"] - macro_bacc if holdout else float("nan")
            ),
        }
    results["unseen"] = unseen_results

    # 3. Robustness ---------------------------------------------------------
    robustness = {}
    grid = cfg["robustness"]["grid"]
    for scope, part in (("holdout", seen), ("unseen", test[test["role"] != "seen"])):
        if part.empty:
            continue
        sample = _sample(part, int(cfg["robustness"]["max_per_label"]), cfg["seed"])
        paths, y, p_clean = list(sample["path"]), sample["label"].to_numpy(), sample["p"].to_numpy()
        clean = binary_metrics(y, p_clean, thr)
        rows = []
        for entry in grid:
            p_pert = calibrate(cache.logits(paths, entry["spec"]))
            m = binary_metrics(y, p_pert, thr)
            rows.append({
                "family": entry.get("family", entry["spec"].split(":")[0]),
                "spec": entry["spec"],
                "severity": entry.get("severity", "mild"),
                "auc": m["auc"],
                "balanced_accuracy": m["balanced_accuracy"],
                "recall": m["recall"],
                "fpr": m["fpr"],
                "auc_drop": clean["auc"] - m["auc"],
                "balanced_accuracy_drop": clean["balanced_accuracy"] - m["balanced_accuracy"],
                **stability(p_clean, p_pert, thr),
            })
        mild = [r for r in rows if r["severity"] == "mild"]
        robustness[scope] = {
            "n": int(len(sample)),
            "clean": clean,
            "rows": rows,
            "worst_mild_auc_drop": max((r["auc_drop"] for r in mild if not math.isnan(r["auc_drop"])), default=float("nan")),
            "worst_mild_flip_rate": max((r["flip_rate"] for r in mild if not math.isnan(r["flip_rate"])), default=float("nan")),
        }
    results["robustness"] = robustness

    # 4. Bias ---------------------------------------------------------------
    min_n = int(cfg["bias"]["min_per_label"])
    bias = {}
    for scope, part in (("holdout", seen), ("all_test", test)):
        if part.empty:
            continue
        bias[scope] = {
            "by_category_group": _group_table(part, "category_group", thr, min_n),
            "by_resolution": _group_table(part, "resolution", thr, min_n),
            "by_source_format": _group_table(part, "fmt", thr, min_n),
        }
    results["bias"] = bias
    results["bias_min_per_label"] = min_n

    # Acceptance ------------------------------------------------------------
    acc = cfg["acceptance"]
    checks = []
    if holdout:
        for key, limit in acc["holdout"].items():
            checks.append(_check(f"hold-out {key}", holdout[key], ">=", limit))
    u = unseen_results["unseen"]
    if u["per_generator"]:
        a = acc["unseen"]
        checks += [
            _check("unseen macro AUC", u["macro_auc"], ">=", a["macro_auc"]),
            _check("unseen macro balanced accuracy", u["macro_balanced_accuracy"], ">=", a["macro_balanced_accuracy"]),
            _check("unseen worst-generator AUC", u["worst_auc"], ">=", a["worst_auc"]),
            _check("AUC drop, hold-out to unseen", u["auc_drop_vs_holdout"], "<=", a["max_auc_drop"]),
        ]
    for scope, r in robustness.items():
        a = acc["robustness_mild"]
        checks += [
            _check(f"worst mild AUC drop ({scope})", r["worst_mild_auc_drop"], "<=", a["max_auc_drop"]),
            _check(f"worst mild label-flip rate ({scope})", r["worst_mild_flip_rate"], "<=", a["max_flip_rate"]),
        ]
    if "holdout" in bias:
        a = acc["fairness"]
        cat = bias["holdout"]["by_category_group"]
        checks += [
            _check("category balanced-accuracy gap (hold-out)", cat["balanced_accuracy_gap"], "<=", a["max_balanced_accuracy_gap"]),
            _check("category false-positive-rate gap (hold-out)", cat["fpr_gap"], "<=", a["max_fpr_gap"]),
        ]
    results["acceptance"] = checks
    verdicts = [c["passed"] for c in checks]
    results["verdict"] = (
        "FAIL" if any(v is False for v in verdicts)
        else "INCOMPLETE" if (not verdicts or any(v is None for v in verdicts))
        else "PASS"
    )
    results["config"] = {k: v for k, v in cfg.items() if k != "robustness"} | {
        "robustness_max_per_label": cfg["robustness"]["max_per_label"]
    }

    from .report import render_report

    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    (out_dir / "REPORT.md").write_text(render_report(results, manifest))
    return results


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
