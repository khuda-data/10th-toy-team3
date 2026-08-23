from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params: dict


def candidates() -> list[Candidate]:
    items: list[Candidate] = []
    for c_value in (0.03, 0.1, 0.3, 1.0):
        items.append(Candidate(f"logit_c{c_value}", "logit", {"C": c_value}))
    for depth in (4, 6, 8, 12):
        for leaf in (30, 50, 100):
            for max_features in ("sqrt", 0.5):
                items.append(
                    Candidate(
                        f"rf_d{depth}_leaf{leaf}_mf{max_features}",
                        "rf",
                        {
                            "n_estimators": 200,
                            "max_depth": depth,
                            "min_samples_leaf": leaf,
                            "max_features": max_features,
                        },
                    )
                )
    items.append(
        Candidate(
            "xgb_regularized",
            "xgb",
            {
                "n_estimators": 400,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 10.0,
                "reg_alpha": 0.0,
            },
        )
    )
    return items


def make_estimator(candidate: Candidate, seed: int):
    if candidate.family == "logit":
        return LogisticRegression(
            C=candidate.params["C"],
            max_iter=3_000,
            solver="liblinear",
            random_state=seed,
        )
    if candidate.family == "rf":
        return RandomForestClassifier(
            **candidate.params,
            class_weight=None,
            n_jobs=1,
            random_state=seed,
        )
    if candidate.family == "xgb":
        return XGBClassifier(
            **candidate.params,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
            tree_method="hist",
        )
    raise ValueError(candidate.family)


def predict_scores(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
