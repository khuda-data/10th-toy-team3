"""Select temperature scaling for the frozen market/model blends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.evaluation.race_metrics import race_metrics, temperature_scale
from src.models.common import PREDICTION_DIR, REPORT_DIR, save_report, utc_now


TEMPERATURE_GRID = (0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.50)
BLEND_POLICY = PROJECT_ROOT / "data" / "manifests" / "market_blend_policy.json"
TEMPERATURE_POLICY = (
    PROJECT_ROOT / "data" / "manifests" / "temperature_policy.json"
)
MODEL_SPECS = {
    "M1_logistic": {
        "input": PREDICTION_DIR / "m1_logistic_calibration_blended.csv.gz",
        "output": PREDICTION_DIR / "m1_logistic_calibration_final.csv.gz",
    },
    "M2_xgboost": {
        "input": PREDICTION_DIR / "m2_xgboost_calibration_blended.csv.gz",
        "output": PREDICTION_DIR / "m2_xgboost_calibration_final.csv.gz",
    },
}


def evaluate_temperature_grid(frame: pd.DataFrame) -> list[dict[str, object]]:
    results = []
    for temperature in TEMPERATURE_GRID:
        probabilities = temperature_scale(
            frame,
            frame["p_blend"].to_numpy(),
            temperature=temperature,
        )
        results.append(
            {"temperature": temperature, "metrics": race_metrics(frame, probabilities)}
        )
    return results


def select_temperature(results: list[dict[str, object]]) -> dict[str, object]:
    """Minimize Log Loss, then Brier, then distance from identity T=1."""
    return min(
        results,
        key=lambda row: (
            row["metrics"]["race_log_loss"],
            row["metrics"]["race_brier"],
            abs(row["temperature"] - 1.0),
            row["temperature"],
        ),
    )


def validate_input(frame: pd.DataFrame, expected_lambda: float) -> None:
    required = {
        "race_id",
        "entry_id",
        "rcDate",
        "win",
        "q_market",
        "p_model_race",
        "blend_lambda",
        "p_blend",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing temperature input columns: {missing}")
    if len(frame) != 6582 or frame["race_id"].nunique() != 641:
        raise ValueError("Temperature selection must use the fixed Calibration fold")
    if not np.allclose(frame["blend_lambda"], expected_lambda, atol=1e-12):
        raise ValueError("Blend lambda does not match the frozen stage-13 policy")
    if not frame.groupby("race_id")["win"].sum().eq(1).all():
        raise ValueError("Calibration contains a race without exactly one winner")
    sums = frame.groupby("race_id")["p_blend"].sum().to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-9):
        raise ValueError("p_blend does not sum to one within every race")


def save_final_predictions(path: Path, frame: pd.DataFrame, selected: dict) -> None:
    temperature = float(selected["temperature"])
    output = frame[
        [
            "race_id",
            "entry_id",
            "rcDate",
            "win",
            "q_market",
            "p_model_race",
            "blend_lambda",
            "p_blend",
        ]
    ].copy()
    output["temperature"] = temperature
    output["p_final"] = temperature_scale(
        frame, frame["p_blend"].to_numpy(), temperature=temperature
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, compression="gzip", encoding="utf-8")


def main() -> int:
    blend_policy = json.loads(BLEND_POLICY.read_text(encoding="utf-8"))
    model_results = {}
    candidates = []

    for model_name, spec in MODEL_SPECS.items():
        frame = pd.read_csv(spec["input"])
        expected_lambda = float(blend_policy["selected_lambdas"][model_name])
        validate_input(frame, expected_lambda)
        grid = evaluate_temperature_grid(frame)
        selected = select_temperature(grid)
        save_final_predictions(spec["output"], frame, selected)
        result = {
            "blend_lambda": expected_lambda,
            "selection_data": "Calibration only; lambda frozen before temperature search",
            "rows": int(len(frame)),
            "races": int(frame["race_id"].nunique()),
            "temperature_grid": grid,
            "selected_temperature": selected["temperature"],
            "selected_metrics": selected["metrics"],
            "identity_temperature_metrics": next(
                row["metrics"] for row in grid if row["temperature"] == 1.0
            ),
            "prediction_path": str(Path(spec["output"]).relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
        }
        model_results[model_name] = result
        candidates.append(
            {
                "model": model_name,
                "lambda": expected_lambda,
                "temperature": selected["temperature"],
                "metrics": selected["metrics"],
            }
        )

    best = min(
        candidates,
        key=lambda row: (
            row["metrics"]["race_log_loss"],
            row["metrics"]["race_brier"],
            abs(row["temperature"] - 1.0),
            row["lambda"],
            row["model"],
        ),
    )
    deployment_candidate = {
        **best,
        "reason": "lowest sequentially calibrated Race Log Loss",
        "status": "candidate_pending_final_test_and_bootstrap",
    }
    report = {
        "experiment": "stage_14_temperature_scaling",
        "created_at": utc_now(),
        "formula": "p_i(T) = softmax(log(p_i) / T) within race",
        "selection_metric": "Calibration Race Log Loss; Race Brier and proximity to T=1 are tie-breakers",
        "caution": "Lambda and temperature were selected sequentially on the same Calibration fold; reported gain may be optimistic.",
        "models": model_results,
        "deployment_candidate": deployment_candidate,
        "test_policy": {"evaluated": False, "reason": "Final test remains sealed."},
    }
    report_path = REPORT_DIR / "stage_14_temperature_scaling.json"
    save_report(report_path, report)

    policy = {
        "policy_version": 1,
        "created_at": report["created_at"],
        "selection_source": "reports/experiments/stage_14_temperature_scaling.json",
        "selection_fold": "calibration",
        "temperature_grid": list(TEMPERATURE_GRID),
        "selected_temperatures": {
            model: details["selected_temperature"]
            for model, details in model_results.items()
        },
        "deployment_candidate": deployment_candidate,
        "sequential_selection_warning": report["caution"],
        "final_test_evaluated": False,
    }
    TEMPERATURE_POLICY.parent.mkdir(parents=True, exist_ok=True)
    TEMPERATURE_POLICY.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
