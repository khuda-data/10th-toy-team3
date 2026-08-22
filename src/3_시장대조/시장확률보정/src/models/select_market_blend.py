"""Select the market/model geometric blend lambda on Calibration races."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.evaluation.race_metrics import geometric_blend, race_metrics
from src.models.common import PREDICTION_DIR, REPORT_DIR, save_report, utc_now


LAMBDA_GRID = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00)
NORMALIZATION_POLICY = (
    PROJECT_ROOT / "data" / "manifests" / "normalization_policy.json"
)
BLEND_POLICY = PROJECT_ROOT / "data" / "manifests" / "market_blend_policy.json"
MODEL_SPECS = {
    "M1_logistic": {
        "input": PREDICTION_DIR / "m1_logistic_calibration_normalized.csv.gz",
        "output": PREDICTION_DIR / "m1_logistic_calibration_blended.csv.gz",
    },
    "M2_xgboost": {
        "input": PREDICTION_DIR / "m2_xgboost_calibration_normalized.csv.gz",
        "output": PREDICTION_DIR / "m2_xgboost_calibration_blended.csv.gz",
    },
}


def evaluate_lambda_grid(frame: pd.DataFrame) -> list[dict[str, object]]:
    results = []
    for lam in LAMBDA_GRID:
        probabilities = geometric_blend(
            frame,
            frame["q_market"].to_numpy(),
            frame["p_model_race"].to_numpy(),
            lam=lam,
        )
        results.append({"lambda": lam, "metrics": race_metrics(frame, probabilities)})
    return results


def select_lambda(results: list[dict[str, object]]) -> dict[str, object]:
    """Minimize Log Loss, then Brier, then prefer the smaller market-safe lambda."""
    return min(
        results,
        key=lambda row: (
            row["metrics"]["race_log_loss"],
            row["metrics"]["race_brier"],
            row["lambda"],
        ),
    )


def validate_input(frame: pd.DataFrame, expected_method: str) -> None:
    required = {"race_id", "entry_id", "rcDate", "win", "q_market", "p_model_race"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing blend input columns: {missing}")
    if len(frame) != 6582 or frame["race_id"].nunique() != 641:
        raise ValueError("Blend selection must use the fixed 641-race Calibration fold")
    if "normalization_method" in frame and not frame["normalization_method"].eq(
        expected_method
    ).all():
        raise ValueError("Prediction normalization does not match the frozen policy")
    if not frame.groupby("race_id")["win"].sum().eq(1).all():
        raise ValueError("Calibration contains a race without exactly one winner")
    for column in ("q_market", "p_model_race"):
        sums = frame.groupby("race_id")[column].sum().to_numpy()
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise ValueError(f"{column} does not sum to one within every race")


def save_blended_predictions(path: Path, frame: pd.DataFrame, selected: dict) -> None:
    lam = float(selected["lambda"])
    output = frame[
        ["race_id", "entry_id", "rcDate", "win", "q_market", "p_model_race"]
    ].copy()
    output["blend_lambda"] = lam
    output["p_blend"] = geometric_blend(
        frame,
        frame["q_market"].to_numpy(),
        frame["p_model_race"].to_numpy(),
        lam=lam,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, compression="gzip", encoding="utf-8")


def main() -> int:
    normalization_policy = json.loads(
        NORMALIZATION_POLICY.read_text(encoding="utf-8")
    )
    model_results = {}
    candidates = []

    for model_name, spec in MODEL_SPECS.items():
        frame = pd.read_csv(spec["input"])
        expected_method = normalization_policy["selected_methods"][model_name]
        validate_input(frame, expected_method)
        grid = evaluate_lambda_grid(frame)
        selected = select_lambda(grid)
        save_blended_predictions(spec["output"], frame, selected)
        result = {
            "normalization_method": expected_method,
            "selection_data": "Calibration only",
            "rows": int(len(frame)),
            "races": int(frame["race_id"].nunique()),
            "lambda_grid": grid,
            "selected_lambda": selected["lambda"],
            "selected_metrics": selected["metrics"],
            "prediction_path": str(Path(spec["output"]).relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
        }
        model_results[model_name] = result
        candidates.append(
            {
                "model": model_name,
                "lambda": selected["lambda"],
                "metrics": selected["metrics"],
            }
        )

    best = min(
        candidates,
        key=lambda row: (
            row["metrics"]["race_log_loss"],
            row["metrics"]["race_brier"],
            row["lambda"],
            row["model"],
        ),
    )
    if best["lambda"] == 0.0:
        deployment_candidate = {
            "model": "M0_market",
            "source_model": best["model"],
            "lambda": 0.0,
            "reason": "lambda=0 makes the blend exactly equal to the market baseline",
            "metrics": best["metrics"],
        }
    else:
        deployment_candidate = {**best, "reason": "lowest Calibration Race Log Loss"}

    report = {
        "experiment": "stage_13_market_geometric_blend",
        "created_at": utc_now(),
        "formula": "p_i proportional to q_market_i^(1-lambda) * p_model_i^lambda",
        "selection_metric": "Calibration Race Log Loss; Race Brier and lower lambda are tie-breakers",
        "models": model_results,
        "deployment_candidate": deployment_candidate,
        "test_policy": {"evaluated": False, "reason": "Final test remains sealed."},
    }
    report_path = REPORT_DIR / "stage_13_market_blend.json"
    save_report(report_path, report)

    policy = {
        "policy_version": 1,
        "created_at": report["created_at"],
        "selection_source": "reports/experiments/stage_13_market_blend.json",
        "selection_fold": "calibration",
        "lambda_grid": list(LAMBDA_GRID),
        "selected_lambdas": {
            model: details["selected_lambda"] for model, details in model_results.items()
        },
        "deployment_candidate": deployment_candidate,
        "final_test_evaluated": False,
    }
    BLEND_POLICY.parent.mkdir(parents=True, exist_ok=True)
    BLEND_POLICY.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
