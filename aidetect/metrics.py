"""Binary detection metrics. The positive class is AI-Generated (label 1)."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

NAN = float("nan")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else NAN


def binary_metrics(y, p, threshold: float = 0.5) -> dict:
    """Threshold metrics at `threshold` plus threshold-free AUC and AP.

    Metrics that are undefined for the given data (for example AUC when only one
    class is present) are returned as NaN rather than raising.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    keep = ~np.isnan(p)
    y, p = y[keep], p[keep]
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    tnr = _safe_div(tn, fp + tn)
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
        f1 = NAN if (tp + fn) == 0 else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    rates = [r for r in (recall, tnr) if not math.isnan(r)]
    both = (tp + fn) > 0 and (fp + tn) > 0
    return {
        "n": int(len(y)),
        "n_real": fp + tn,
        "n_fake": tp + fn,
        "accuracy": _safe_div(tp + tn, len(y)),
        "balanced_accuracy": float(np.mean(rates)) if rates else NAN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "auc": float(roc_auc_score(y, p)) if both else NAN,
        "ap": float(average_precision_score(y, p)) if both else NAN,
    }


def bootstrap_ci(
    y,
    p,
    threshold: float = 0.5,
    keys=("precision", "recall", "f1", "auc"),
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap confidence intervals, resampling within each class."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    keep = ~np.isnan(p)
    y, p = y[keep], p[keep]
    rng = np.random.default_rng(seed)
    idx_real = np.flatnonzero(y == 0)
    idx_fake = np.flatnonzero(y == 1)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        parts = []
        if len(idx_real):
            parts.append(rng.choice(idx_real, len(idx_real), replace=True))
        if len(idx_fake):
            parts.append(rng.choice(idx_fake, len(idx_fake), replace=True))
        if not parts:
            break
        idx = np.concatenate(parts)
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    out = {}
    for k, values in samples.items():
        values = [v for v in values if not math.isnan(v)]
        if values:
            lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
            out[k] = (float(lo), float(hi))
        else:
            out[k] = (NAN, NAN)
    return out


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    """ECE of the predicted AI probability against the observed AI frequency."""
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    keep = ~np.isnan(p)
    y, p = y[keep], p[keep]
    if len(y) == 0:
        return NAN
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def tpr_at_fpr(y, p, target_fpr: float = 0.01) -> float:
    """Recall on generated images at the threshold giving `target_fpr` on real ones."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    real, fake = p[(y == 0) & ~np.isnan(p)], p[(y == 1) & ~np.isnan(p)]
    if len(real) == 0 or len(fake) == 0:
        return NAN
    thr = np.quantile(real, 1 - target_fpr, method="higher")
    return float((fake > thr).mean())


def stability(p_clean, p_perturbed, threshold: float = 0.5) -> dict:
    """How much predictions move when the same images are perturbed."""
    a = np.asarray(p_clean, dtype=float)
    b = np.asarray(p_perturbed, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return {"flip_rate": NAN, "mean_abs_prob_change": NAN}
    return {
        "flip_rate": float(((a >= threshold) != (b >= threshold)).mean()),
        "mean_abs_prob_change": float(np.abs(a - b).mean()),
    }
