"""Run the one-time Final Test evaluation using only pre-frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries
from src.data.validate_schema import sha256_file
from src.evaluation.race_metrics import (
    geometric_blend,
    normalize_model_probabilities,
    race_metrics,
    temperature_scale,
)
from src.features.preprocess import model_frame
from src.models.common import PREDICTION_DIR, REPORT_DIR, save_report, utc_now


FREEZE_PATH = PROJECT_ROOT / "data" / "manifests" / "pre_final_test_freeze.json"
RESULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "final_test_evaluation.json"
REPORT_PATH = REPORT_DIR / "stage_15_final_test.json"
MODEL_SPECS = {
    "M1_logistic": {
        "artifact": PROJECT_ROOT / "artifacts" / "models" / "m1_logistic.joblib",
        "prediction": PREDICTION_DIR / "m1_logistic_test_final.csv.gz",
    },
    "M2_xgboost": {
        "artifact": PROJECT_ROOT / "artifacts" / "models" / "m2_xgboost.joblib",
        "prediction": PREDICTION_DIR / "m2_xgboost_test_final.csv.gz",
    },
}


def verify_freeze() -> dict:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("Pre-test freeze manifest is required")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze["final_test_opened"]:
        raise ValueError("Freeze manifest must have final_test_opened=false")
    mismatches = []
    for item in freeze["files"]:
        path = PROJECT_ROOT / item["path"]
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != item["sha256"]:
            mismatches.append(
                {"path": item["path"], "expected": item["sha256"], "actual": actual}
            )
    if mismatches:
        raise ValueError(f"Frozen input checksum mismatch: {mismatches}")
    return freeze


def guard_single_evaluation() -> None:
    existing = [path for path in (REPORT_PATH, RESULT_MANIFEST) if path.exists()]
    existing.extend(
        spec["prediction"] for spec in MODEL_SPECS.values() if spec["prediction"].exists()
    )
    if existing:
        raise FileExistsError(
            "Final Test outputs already exist; automatic re-evaluation is forbidden: "
            + ", ".join(str(path) for path in existing)
        )


def load_policies() -> tuple[dict, dict, dict]:
    paths = (
        "data/manifests/normalization_policy.json",
        "data/manifests/market_blend_policy.json",
        "data/manifests/temperature_policy.json",
    )
    return tuple(
        json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8")) for path in paths
    )


def save_test_predictions(path: Path, frame: pd.DataFrame, values: dict) -> None:
    output = frame[["race_id", "entry_id", "rcDate", "win", "q_market"]].copy()
    for name, value in values.items():
        output[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, compression="gzip", encoding="utf-8")


def main() -> int:
    guard_single_evaluation()
    freeze = verify_freeze()
    normalization, blend, temperature = load_policies()

    # This is the first and only point at which the sealed Test fold is loaded.
    test = load_model_entries(("test",))
    if len(test) != 6639 or test["race_id"].nunique() != 635:
        raise ValueError("Final Test fold shape differs from the frozen contract")
    if not test.groupby("race_id")["win"].sum().eq(1).all():
        raise ValueError("Final Test requires exactly one winner per race")

    market_metrics = race_metrics(test, test["q_market"].to_numpy())
    model_results = {}
    for model_name, spec in MODEL_SPECS.items():
        bundle = joblib.load(spec["artifact"])
        raw = bundle["estimator"].predict_proba(
            model_frame(test, bundle["feature_schema"])
        )[:, 1]
        method = normalization["selected_methods"][model_name]
        model_probability = normalize_model_probabilities(
            test, raw, method=method
        )
        lam = float(blend["selected_lambdas"][model_name])
        blended = geometric_blend(
            test,
            test["q_market"].to_numpy(),
            model_probability,
            lam=lam,
        )
        temp = float(temperature["selected_temperatures"][model_name])
        final = temperature_scale(test, blended, temperature=temp)
        metrics = {
            "standalone_model": race_metrics(test, model_probability),
            "market_blend": race_metrics(test, blended),
            "temperature_scaled_final": race_metrics(test, final),
        }
        final_metrics = metrics["temperature_scaled_final"]
        model_results[model_name] = {
            "normalization_method": method,
            "lambda": lam,
            "temperature": temp,
            "metrics": metrics,
            "market_deltas_positive_is_better": {
                "race_log_loss": market_metrics["race_log_loss"]
                - final_metrics["race_log_loss"],
                "race_brier": market_metrics["race_brier"]
                - final_metrics["race_brier"],
                "top1_accuracy": final_metrics["top1_accuracy"]
                - market_metrics["top1_accuracy"],
            },
            "prediction_path": str(Path(spec["prediction"]).relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
        }
        save_test_predictions(
            spec["prediction"],
            test,
            {
                "p_model_raw": raw,
                "normalization_method": method,
                "p_model_race": model_probability,
                "blend_lambda": np.full(len(test), lam),
                "p_blend": blended,
                "temperature": np.full(len(test), temp),
                "p_final": final,
            },
        )

    frozen_candidate = freeze["frozen_deployment_candidate"]
    primary_model = frozen_candidate["model"]
    primary = model_results[primary_model]
    deltas = primary["market_deltas_positive_is_better"]
    primary_conclusion = {
        "model": primary_model,
        "beats_market_on_race_log_loss": deltas["race_log_loss"] > 0,
        "beats_market_on_race_brier": deltas["race_brier"] > 0,
        "beats_market_on_both_primary_metrics": (
            deltas["race_log_loss"] > 0 and deltas["race_brier"] > 0
        ),
        "statistical_significance_assessed": False,
        "next_required_check": "race-level paired bootstrap in stage 16",
    }
    report = {
        "experiment": "stage_15_single_final_test",
        "evaluated_at": utc_now(),
        "freeze_manifest": "data/manifests/pre_final_test_freeze.json",
        "test_fold": freeze["test_fold"],
        "frozen_deployment_candidate": frozen_candidate,
        "market_baseline": market_metrics,
        "models": model_results,
        "primary_conclusion": primary_conclusion,
        "post_test_policy": "No model, feature, lambda, temperature, or normalization changes are permitted based on this Test result.",
        "test_policy": {"evaluated": True, "evaluation_count": 1},
    }
    save_report(REPORT_PATH, report)

    result_files = [REPORT_PATH] + [spec["prediction"] for spec in MODEL_SPECS.values()]
    result_manifest = {
        "manifest_version": 1,
        "evaluated_at": report["evaluated_at"],
        "freeze_manifest_path": "data/manifests/pre_final_test_freeze.json",
        "freeze_manifest_sha256": sha256_file(FREEZE_PATH),
        "evaluation_count": 1,
        "final_test_evaluated": True,
        "frozen_deployment_candidate": frozen_candidate,
        "primary_conclusion": primary_conclusion,
        "outputs": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in result_files
        ],
    }
    RESULT_MANIFEST.write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(primary_conclusion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
