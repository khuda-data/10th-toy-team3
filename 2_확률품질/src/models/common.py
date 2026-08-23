"""Shared artifact, prediction, and walk-forward helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import make_walk_forward_folds
from src.evaluation.race_metrics import normalize_by_race, race_metrics
from src.features.preprocess import FeatureSchema, model_frame


REPORT_DIR = PROJECT_ROOT / "reports" / "experiments"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"
PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_estimator(estimator, frame: pd.DataFrame, schema: FeatureSchema):
    raw = estimator.predict_proba(model_frame(frame, schema))[:, 1]
    normalized = normalize_by_race(frame, raw)
    return race_metrics(frame, normalized), raw, normalized


def walk_forward_evaluate(train, schema, estimator_factory) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for fold in make_walk_forward_folds(train):
        fit_frame = train.loc[fold["train_index"]]
        valid_frame = train.loc[fold["valid_index"]]
        estimator = estimator_factory()
        estimator.fit(model_frame(fit_frame, schema), fit_frame["win"].to_numpy())
        metrics, _, _ = evaluate_estimator(estimator, valid_frame, schema)
        results.append(
            {
                "fold": fold["fold"],
                "train_rows": int(len(fit_frame)),
                "train_races": int(fit_frame["race_id"].nunique()),
                "train_date_max": fold["train_date_max"],
                "valid_rows": int(len(valid_frame)),
                "valid_races": int(valid_frame["race_id"].nunique()),
                "valid_date_min": fold["valid_date_min"],
                "valid_date_max": fold["valid_date_max"],
                "metrics": metrics,
            }
        )
    return results


def save_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def save_predictions(
    path: Path,
    frame: pd.DataFrame,
    raw: np.ndarray,
    normalized: np.ndarray,
    *,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame[["race_id", "entry_id", "rcDate", "win", "q_market"]].copy()
    output["model"] = model_name
    output["p_model_raw"] = raw
    output["p_model_race"] = normalized
    output.to_csv(path, index=False, compression="gzip", encoding="utf-8")
