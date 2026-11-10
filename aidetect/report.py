"""Render evaluation results as the Markdown evaluation report."""

from __future__ import annotations

import math

import pandas as pd


def _f(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No data._\n"
    head = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_f(r.get(key)) for key, _ in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body]) + "\n"


_GEN_COLS = [
    ("generator", "Generator"), ("n_real", "Real"), ("n_fake", "Generated"),
    ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
    ("fpr", "FPR"), ("balanced_accuracy", "Bal. acc."), ("auc", "AUC"),
]
_GROUP_COLS = [
    ("group", "Group"), ("n_real", "Real"), ("n_fake", "Generated"),
    ("balanced_accuracy", "Bal. acc."), ("recall", "Recall"), ("fpr", "FPR"), ("auc", "AUC"),
]
_SCOPE_TITLES = {"holdout": "Hold-out (seen generators)", "unseen": "Unseen and near-seen generators", "all_test": "All test images"}
_ROBUST_COLS = [
    ("spec", "Perturbation"), ("severity", "Severity"), ("auc", "AUC"), ("auc_drop", "AUC drop"),
    ("balanced_accuracy", "Bal. acc."), ("recall", "Recall"), ("fpr", "FPR"),
    ("flip_rate", "Label flips"), ("mean_abs_prob_change", "Mean prob. change"),
]


def render_report(results: dict, manifest: pd.DataFrame | None = None) -> str:
    out: list[str] = []
    w = out.append
    thr = results["threshold"]
    w("# Evaluation report\n")
    w(
        f"Model `{results['model']}`. Decision threshold {_f(thr)} on the calibrated AI probability, "
        f"fixed before testing. {results['n_test_images']} test images scored"
        + (f", {results['n_unreadable']} unreadable files skipped" if results["n_unreadable"] else "")
        + ". Positive class: AI-Generated.\n"
    )

    w(f"## Acceptance: {results['verdict']}\n")
    rows = [
        {
            "check": c["check"],
            "value": c["value"],
            "target": f"{c['op']} {_f(c['limit'])}",
            "result": {True: "pass", False: "FAIL", None: "not measurable"}[c["passed"]],
        }
        for c in results["acceptance"]
    ]
    w(_table(rows, [("check", "Check"), ("value", "Value"), ("target", "Target"), ("result", "Result")]))
    if results["verdict"] == "INCOMPLETE":
        w("Some checks could not be measured (missing data for that slice); see the sections below.\n")

    # 1 -------------------------------------------------------------------
    w("## 1. Hold-out test set (seen generators)\n")
    h = results.get("holdout")
    if not h:
        w("_The manifest has no test images from seen generators._\n")
    else:
        ci = h.get("ci95", {})
        rows = []
        for key, title in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("auc", "AUC")):
            lo, hi = ci.get(key, (float("nan"), float("nan")))
            rows.append({"metric": title, "value": h[key], "ci": f"{_f(lo)} to {_f(hi)}"})
        for key, title in (
            ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
            ("fpr", "False-positive rate"), ("ap", "Average precision"),
            ("tpr_at_1pct_fpr", "Recall at 1% FPR"), ("ece", "Expected calibration error"),
        ):
            rows.append({"metric": title, "value": h[key], "ci": ""})
        w(f"{h['n_real']} real and {h['n_fake']} generated images.\n")
        w(_table(rows, [("metric", "Metric"), ("value", "Value"), ("ci", "95% CI (bootstrap)")]))
        w("By generator:\n")
        w(_table(h.get("per_generator", []), _GEN_COLS))

    # 2 -------------------------------------------------------------------
    w("## 2. Generators not used in training\n")
    for role, title in (
        ("unseen", "Unseen generators"),
        ("near_seen", "Near-seen generators (same family as a training generator)"),
    ):
        u = results["unseen"].get(role, {})
        w(f"### {title}\n")
        if not u.get("per_generator"):
            w("_None in the manifest._\n")
            continue
        w(
            f"Macro average over generators: AUC {_f(u['macro_auc'])}, balanced accuracy "
            f"{_f(u['macro_balanced_accuracy'])}, recall {_f(u['macro_recall'])}, FPR {_f(u['macro_fpr'])}. "
            f"Change from hold-out: AUC {_f(-u['auc_drop_vs_holdout'])}, balanced accuracy "
            f"{_f(-u['balanced_accuracy_drop_vs_holdout'])}. Worst generator AUC {_f(u['worst_auc'])}.\n"
        )
        w(_table(u["per_generator"], [("family", "Family"), *_GEN_COLS]))

    # 3 -------------------------------------------------------------------
    w("## 3. Robustness to image processing\n")
    w(
        "Each row re-scores the same sample after one perturbation. Drops are relative to the clean "
        "sample. Label flips is the share of images whose predicted label changed.\n"
    )
    if not results["robustness"]:
        w("_Not run._\n")
    for scope, r in results["robustness"].items():
        c = r["clean"]
        w(f"### {_SCOPE_TITLES.get(scope, scope)} sample ({r['n']} images)\n")
        w(
            f"Clean: AUC {_f(c['auc'])}, balanced accuracy {_f(c['balanced_accuracy'])}, "
            f"recall {_f(c['recall'])}, FPR {_f(c['fpr'])}. Worst mild condition: AUC drop "
            f"{_f(r['worst_mild_auc_drop'])}, label flips {_f(r['worst_mild_flip_rate'])}.\n"
        )
        w(_table(r["rows"], _ROBUST_COLS))

    # 4 -------------------------------------------------------------------
    w("## 4. Bias analysis\n")
    w(
        f"A group needs at least {results.get('bias_min_per_label', 50)} images of a class before the "
        "metrics that depend on that class are reported; smaller groups show n/a. "
        "A gap is the largest minus the smallest value across groups.\n"
    )
    names = {
        "by_category_group": "By content category",
        "by_resolution": "By resolution (short side, pixels)",
        "by_source_format": "By source file format",
    }
    for scope, tables in results["bias"].items():
        w(f"### {_SCOPE_TITLES.get(scope, scope)}\n")
        for key, title in names.items():
            t = tables[key]
            w(
                f"**{title}.** Category known for {_f(t['coverage'], 2)} of images. Gaps: balanced accuracy "
                f"{_f(t['balanced_accuracy_gap'])}, recall {_f(t['recall_gap'])}, FPR {_f(t['fpr_gap'])}.\n"
                if key == "by_category_group"
                else f"**{title}.** Gaps: balanced accuracy {_f(t['balanced_accuracy_gap'])}, "
                f"recall {_f(t['recall_gap'])}, FPR {_f(t['fpr_gap'])}.\n"
            )
            w(_table(t["rows"], _GROUP_COLS))

    # Appendix --------------------------------------------------------------
    if manifest is not None and len(manifest):
        w("## Appendix: data used\n")
        counts = (
            manifest.assign(kind=manifest["label"].map({0: "real", 1: "generated"}))
            .groupby(["split", "role", "generator", "kind"]).size().unstack("kind", fill_value=0)
            .reset_index()
        )
        for col in ("real", "generated"):
            if col not in counts.columns:
                counts[col] = 0
        w(_table(
            counts.to_dict(orient="records"),
            [("split", "Split"), ("role", "Role"), ("generator", "Generator"), ("real", "Real"), ("generated", "Generated")],
        ))
    return "\n".join(out)
