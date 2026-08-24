from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import PERCENTILES


def top_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    take = n if fraction >= 1 else max(1, int(np.floor(n * fraction)))
    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:take]] = True
    return mask


def lift_at(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    y = np.asarray(y_true, dtype=float)
    base = float(y.mean())
    if base == 0:
        return float("nan")
    return float(y[top_mask(scores, fraction)].mean() / base)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    result = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }
    for fraction in PERCENTILES:
        key = f"lift_{int(fraction * 100)}pct"
        result[key] = lift_at(y, p, fraction)
        result[f"hit_rate_{int(fraction * 100)}pct"] = float(
            y[top_mask(p, fraction)].mean()
        )
    return result


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)

    @staticmethod
    def _logit(scores: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        self.model.fit(self._logit(scores), np.asarray(y_true, dtype=int))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(scores))[:, 1]


def metrics_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records).sort_values(
        ["target", "feature_set", "lift_10pct", "pr_auc"],
        ascending=[True, True, False, False],
    )
