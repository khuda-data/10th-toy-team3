"""Train and evaluate the M2 market-independent XGBoost baseline."""

from __future__ import annotations

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.model_data import load_model_entries
from src.features.preprocess import infer_feature_schema, make_preprocessor, model_frame
from src.models.common import (
    ARTIFACT_DIR,
    PREDICTION_DIR,
    REPORT_DIR,
    evaluate_estimator,
    save_predictions,
    save_report,
    utc_now,
    walk_forward_evaluate,
)


PARAMETERS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "n_jobs": -1,
    "random_state": 42,
}


def make_estimator(schema):
    return Pipeline(
        [
            ("preprocess", make_preprocessor(schema, scale_numeric=False)),
            ("model", XGBClassifier(**PARAMETERS)),
        ]
    )


def main() -> int:
    train = load_model_entries(("train",))
    calibration = load_model_entries(("calibration",))
    schema = infer_feature_schema(train)
    cv = walk_forward_evaluate(train, schema, lambda: make_estimator(schema))

    estimator = make_estimator(schema)
    estimator.fit(model_frame(train, schema), train["win"].to_numpy())
    train_metrics, _, _ = evaluate_estimator(estimator, train, schema)
    cal_metrics, raw, normalized = evaluate_estimator(estimator, calibration, schema)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "m2_xgboost.joblib"
    joblib.dump({"estimator": estimator, "feature_schema": schema}, artifact_path)
    save_predictions(
        PREDICTION_DIR / "m2_xgboost_calibration.csv.gz",
        calibration,
        raw,
        normalized,
        model_name="M2_xgboost",
    )
    report = {
        "experiment": "M2_xgboost",
        "created_at": utc_now(),
        "feature_schema": schema.to_dict(),
        "preprocessing_fit_scope": "each walk-forward train fold; final artifact on full train only",
        "hyperparameters": PARAMETERS,
        "walk_forward": cv,
        "evaluation": {"train": train_metrics, "calibration": cal_metrics},
        "artifacts": {
            "model": "artifacts/models/m2_xgboost.joblib",
            "calibration_predictions": "data/predictions/m2_xgboost_calibration.csv.gz",
        },
        "test_policy": {"evaluated": False, "reason": "Final test remains sealed."},
    }
    save_report(REPORT_DIR / "m2_xgboost.json", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
