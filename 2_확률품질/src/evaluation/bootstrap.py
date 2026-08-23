"""Race-level paired bootstrap for frozen Final Test predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file


N_BOOTSTRAP = 5000
RANDOM_SEED = 42
FINAL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "final_test_evaluation.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_16_bootstrap.json"
REPLICATE_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "stage_16_bootstrap_replicates.csv.gz"
)
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "bootstrap_evaluation.json"
MODEL_FILES = {
    "M1_logistic": PROJECT_ROOT / "data" / "predictions" / "m1_logistic_test_final.csv.gz",
    "M2_xgboost": PROJECT_ROOT / "data" / "predictions" / "m2_xgboost_test_final.csv.gz",
}


def verify_final_outputs() -> dict:
    manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    if manifest["evaluation_count"] != 1 or not manifest["final_test_evaluated"]:
        raise ValueError("Bootstrap requires the locked one-time Final Test evaluation")
    for item in manifest["outputs"]:
        path = PROJECT_ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Final Test output checksum mismatch: {item['path']}")
    return manifest


def race_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-race market-minus-model contributions; positive favors model."""
    required = {"race_id", "win", "q_market", "p_final"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing bootstrap columns: {missing}")
    if not frame.groupby("race_id")["win"].sum().eq(1).all():
        raise ValueError("Bootstrap input requires exactly one winner per race")

    working = frame[["race_id", "win", "q_market", "p_final"]].copy()
    winner = working.loc[working["win"].eq(1)].set_index("race_id")
    logloss_delta = -np.log(winner["q_market"].clip(1e-15, 1.0))
    logloss_delta -= -np.log(winner["p_final"].clip(1e-15, 1.0))

    working["market_sq"] = (working["q_market"] - working["win"]) ** 2
    working["model_sq"] = (working["p_final"] - working["win"]) ** 2
    grouped = working.groupby("race_id", sort=True)
    brier_delta = grouped["market_sq"].sum() - grouped["model_sq"].sum()

    working["market_rank"] = grouped["q_market"].rank(method="min", ascending=False)
    working["model_rank"] = grouped["p_final"].rank(method="min", ascending=False)
    winner_rank = working.loc[working["win"].eq(1)].set_index("race_id")
    top1_delta = winner_rank["model_rank"].eq(1).astype(float)
    top1_delta -= winner_rank["market_rank"].eq(1).astype(float)

    result = pd.DataFrame(
        {
            "delta_logloss": logloss_delta,
            "delta_brier": brier_delta,
            "delta_top1": top1_delta,
        }
    ).sort_index()
    if len(result) != 635:
        raise ValueError(f"Expected 635 Test races, found {len(result)}")
    return result


def bootstrap_means(
    values: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return values[sample_indices].mean(axis=1)


def summarize_delta(point_estimate: float, replicates: np.ndarray) -> dict[str, object]:
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    probability_positive = float(np.mean(replicates > 0.0))
    return {
        "direction": "market_loss - model_loss; positive favors model",
        "point_estimate": float(point_estimate),
        "bootstrap_mean": float(replicates.mean()),
        "bootstrap_standard_error": float(replicates.std(ddof=1)),
        "ci_95_percentile": {"lower": float(lower), "upper": float(upper)},
        "probability_model_better": probability_positive,
        "probability_market_equal_or_better": float(1.0 - probability_positive),
        "ci_lower_above_zero": bool(lower > 0.0),
    }


def main() -> int:
    if REPORT_PATH.exists() or MANIFEST_PATH.exists() or REPLICATE_PATH.exists():
        raise FileExistsError("Bootstrap outputs already exist and will not be overwritten")
    final_manifest = verify_final_outputs()
    frames = {name: pd.read_csv(path) for name, path in MODEL_FILES.items()}
    contributions = {name: race_contributions(frame) for name, frame in frames.items()}
    reference_ids = contributions["M2_xgboost"].index
    if any(not part.index.equals(reference_ids) for part in contributions.values()):
        raise ValueError("Model prediction files do not contain identical ordered races")

    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = rng.integers(
        0, len(reference_ids), size=(N_BOOTSTRAP, len(reference_ids))
    )
    replicate_frame = pd.DataFrame({"replicate": np.arange(1, N_BOOTSTRAP + 1)})
    model_results = {}
    for model_name, part in contributions.items():
        metric_results = {}
        for metric in ("logloss", "brier", "top1"):
            column = f"delta_{metric}"
            values = part[column].to_numpy(dtype=float)
            replicates = bootstrap_means(values, sample_indices)
            replicate_frame[f"{model_name}_{column}"] = replicates
            metric_results[column] = summarize_delta(values.mean(), replicates)
        model_results[model_name] = {
            "races": int(len(part)),
            "metrics": metric_results,
        }

    REPLICATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    replicate_frame.to_csv(
        REPLICATE_PATH, index=False, compression="gzip", encoding="utf-8"
    )
    primary_model = final_manifest["frozen_deployment_candidate"]["model"]
    primary = model_results[primary_model]["metrics"]
    primary_conclusion = {
        "model": primary_model,
        "race_log_loss_ci_lower_above_zero": primary["delta_logloss"][
            "ci_lower_above_zero"
        ],
        "race_brier_ci_lower_above_zero": primary["delta_brier"][
            "ci_lower_above_zero"
        ],
        "statistically_supported_on_both_primary_metrics": (
            primary["delta_logloss"]["ci_lower_above_zero"]
            and primary["delta_brier"]["ci_lower_above_zero"]
        ),
        "interpretation": "A positive point estimate without a positive CI lower bound is insufficient evidence of stable market outperformance.",
    }
    report = {
        "experiment": "stage_16_race_level_paired_bootstrap",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": "data/manifests/final_test_evaluation.json",
        "method": {
            "unit": "race_id",
            "paired": True,
            "replacement": True,
            "n_races": 635,
            "n_bootstrap": N_BOOTSTRAP,
            "random_seed": RANDOM_SEED,
            "confidence_interval": "95% percentile",
        },
        "models": model_results,
        "primary_conclusion": primary_conclusion,
        "replicate_path": str(REPLICATE_PATH.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "post_test_policy": "Descriptive inference only; no tuning or model selection from Final Test bootstrap results.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "manifest_version": 1,
        "created_at": report["created_at"],
        "input_final_test_manifest_sha256": sha256_file(FINAL_MANIFEST),
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "primary_conclusion": primary_conclusion,
        "outputs": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (REPORT_PATH, REPLICATE_PATH)
        ],
        "final_test_model_changed": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(primary_conclusion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
