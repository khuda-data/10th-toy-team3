"""Select race-probability normalization using Train walk-forward OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries, make_walk_forward_folds
from src.evaluation.race_metrics import (
    NORMALIZATION_METHODS,
    normalize_model_probabilities,
    race_metrics,
)
from src.features.preprocess import infer_feature_schema, model_frame
from src.models.common import PREDICTION_DIR, REPORT_DIR, save_report, utc_now
from src.models.train_m1_logistic import make_estimator as make_m1
from src.models.train_m2_xgboost import make_estimator as make_m2


POLICY_PATH = PROJECT_ROOT / "data" / "manifests" / "normalization_policy.json"
MODEL_SPECS = {
    "M1_logistic": {
        "factory": make_m1,
        "artifact": PROJECT_ROOT / "artifacts" / "models" / "m1_logistic.joblib",
        "prediction": PREDICTION_DIR / "m1_logistic_calibration_normalized.csv.gz",
    },
    "M2_xgboost": {
        "factory": make_m2,
        "artifact": PROJECT_ROOT / "artifacts" / "models" / "m2_xgboost.joblib",
        "prediction": PREDICTION_DIR / "m2_xgboost_calibration_normalized.csv.gz",
    },
}


def normalization_metrics(frame: pd.DataFrame, raw: np.ndarray) -> dict[str, dict]:
    return {
        method: race_metrics(
            frame,
            normalize_model_probabilities(frame, raw, method=method),
        )
        for method in NORMALIZATION_METHODS
    }


def select_method(metrics: dict[str, dict]) -> str:
    """Select by Log Loss, then Brier, with simpler sum normalization as final tie-break."""
    preference = {"sum": 0, "logit_softmax": 1}
    return min(
        NORMALIZATION_METHODS,
        key=lambda method: (
            metrics[method]["race_log_loss"],
            metrics[method]["race_brier"],
            preference[method],
        ),
    )


def build_oof_predictions(train, schema, estimator_factory):
    frames = []
    fold_results = []
    for fold in make_walk_forward_folds(train):
        fit_frame = train.loc[fold["train_index"]]
        valid_frame = train.loc[fold["valid_index"]].copy()
        estimator = estimator_factory(schema)
        estimator.fit(model_frame(fit_frame, schema), fit_frame["win"].to_numpy())
        raw = estimator.predict_proba(model_frame(valid_frame, schema))[:, 1]
        valid_frame["p_model_raw"] = raw
        fold_results.append(
            {
                "fold": fold["fold"],
                "train_date_max": fold["train_date_max"],
                "valid_date_min": fold["valid_date_min"],
                "valid_date_max": fold["valid_date_max"],
                "metrics": normalization_metrics(valid_frame, raw),
            }
        )
        frames.append(valid_frame)
    return pd.concat(frames, ignore_index=True), fold_results


def save_calibration_predictions(path, frame, raw, selected_method):
    output = frame[["race_id", "entry_id", "rcDate", "win", "q_market"]].copy()
    output["p_model_raw"] = raw
    for method in NORMALIZATION_METHODS:
        output[f"p_{method}"] = normalize_model_probabilities(
            frame, raw, method=method
        )
    output["normalization_method"] = selected_method
    output["p_model_race"] = output[f"p_{selected_method}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, compression="gzip", encoding="utf-8")


def main() -> int:
    train = load_model_entries(("train",))
    calibration = load_model_entries(("calibration",))
    schema = infer_feature_schema(train)
    model_results = {}

    for model_name, spec in MODEL_SPECS.items():
        oof, folds = build_oof_predictions(train, schema, spec["factory"])
        oof_metrics = normalization_metrics(oof, oof["p_model_raw"].to_numpy())
        selected = select_method(oof_metrics)

        bundle = joblib.load(spec["artifact"])
        raw_cal = bundle["estimator"].predict_proba(
            model_frame(calibration, bundle["feature_schema"])
        )[:, 1]
        calibration_metrics = normalization_metrics(calibration, raw_cal)
        save_calibration_predictions(
            spec["prediction"], calibration, raw_cal, selected
        )
        model_results[model_name] = {
            "selection_data": "Train walk-forward OOF only",
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
            "folds": folds,
            "oof_metrics": oof_metrics,
            "selected_method": selected,
            "calibration_diagnostics_not_used_for_selection": calibration_metrics,
            "calibration_prediction_path": str(
                Path(spec["prediction"]).relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
        }

    report = {
        "experiment": "stage_12_race_normalization",
        "created_at": utc_now(),
        "selection_metric": "Train OOF Race Log Loss; Race Brier and sum method are tie-breakers",
        "candidate_methods": {
            "sum": "p_raw / race sum(p_raw)",
            "logit_softmax": "softmax(logit(p_raw)) within race",
        },
        "models": model_results,
        "test_policy": {"evaluated": False, "reason": "Final test remains sealed."},
    }
    report_path = REPORT_DIR / "stage_12_normalization.json"
    save_report(report_path, report)

    policy = {
        "policy_version": 1,
        "created_at": report["created_at"],
        "selection_source": "reports/experiments/stage_12_normalization.json",
        "selection_metric": report["selection_metric"],
        "selected_methods": {
            model: details["selected_method"] for model, details in model_results.items()
        },
        "final_test_evaluated": False,
    }
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
