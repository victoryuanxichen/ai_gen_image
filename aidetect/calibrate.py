"""Turn raw detector scores into probabilities and pick an operating threshold."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    """Platt scaling: ``p = sigmoid(a * logit + b)``. Identity when a=1, b=0."""

    a: float = 1.0
    b: float = 0.0

    def __call__(self, logits) -> np.ndarray:
        return expit(self.a * np.asarray(logits, dtype=float) + self.b)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibrator":
        d = d or {}
        return cls(a=float(d.get("a", 1.0)), b=float(d.get("b", 0.0)))

    @classmethod
    def fit(cls, logits, y) -> "Calibrator":
        """Fit on held-out validation scores. Falls back to identity if the fit
        is impossible (one class only) or would reverse the score ordering."""
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y).astype(int)
        keep = ~np.isnan(logits)
        logits, y = logits[keep], y[keep]
        if len(np.unique(y)) < 2:
            return cls()
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logits.reshape(-1, 1), y)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
            return cls()
        return cls(a=a, b=b)


def threshold_for_fpr(p_real, target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on `p_real` is <= target."""
    p_real = np.asarray(p_real, dtype=float)
    p_real = p_real[~np.isnan(p_real)]
    if len(p_real) == 0:
        return 0.5
    q = float(np.quantile(p_real, 1 - target_fpr, method="higher"))
    # Predictions use p >= threshold, so step just above the quantile value.
    return float(min(np.nextafter(q, 1.0), 1.0))
