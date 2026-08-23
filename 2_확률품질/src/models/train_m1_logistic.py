"""Train and evaluate the M1 market-independent logistic baseline."""

from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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


def make_estimator(schema):
    return Pipeline(
        [
            ("preprocess", make_preprocessor(schema, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    C=0.1, class_weight=None, max_iter=3000
                ),
            ),
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
    artifact_path = ARTIFACT_DIR / "m1_logistic.joblib"
    joblib.dump({"estimator": estimator, "feature_schema": schema}, artifact_path)
    save_predictions(
        PREDICTION_DIR / "m1_logistic_calibration.csv.gz",
        calibration,
        raw,
        normalized,
        model_name="M1_logistic",
    )
    report = {
        "experiment": "M1_logistic",
        "created_at": utc_now(),
        "feature_schema": schema.to_dict(),
        "preprocessing_fit_scope": "each walk-forward train fold; final artifact on full train only",
        "hyperparameters": {
            "penalty": "l2",
            "C": 0.1,
            "class_weight": None,
            "max_iter": 3000,
        },
        "walk_forward": cv,
        "evaluation": {"train": train_metrics, "calibration": cal_metrics},
        "artifacts": {
            "model": "artifacts/models/m1_logistic.joblib",
            "calibration_predictions": "data/predictions/m1_logistic_calibration.csv.gz",
        },
        "test_policy": {"evaluated": False, "reason": "Final test remains sealed."},
    }
    save_report(REPORT_DIR / "m1_logistic.json", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
