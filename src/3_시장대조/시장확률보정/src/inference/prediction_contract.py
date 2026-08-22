"""Generate validated race predictions from pre-race features and odds snapshots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file
from src.evaluation.race_metrics import (
    geometric_blend,
    normalize_model_probabilities,
    temperature_scale,
)
from src.features.preprocess import model_frame


MODEL_ARTIFACT = PROJECT_ROOT / "artifacts" / "models" / "m2_xgboost.joblib"
NORMALIZATION_POLICY = PROJECT_ROOT / "data" / "manifests" / "normalization_policy.json"
BLEND_POLICY = PROJECT_ROOT / "data" / "manifests" / "market_blend_policy.json"
TEMPERATURE_POLICY = PROJECT_ROOT / "data" / "manifests" / "temperature_policy.json"
BETTING_POLICY = PROJECT_ROOT / "data" / "manifests" / "betting_policy.json"
SCHEMA_PATH = PROJECT_ROOT / "data" / "manifests" / "prediction_output_schema.json"

MODEL_NAME = "M2_xgboost"
MODEL_VERSION = "m2_xgboost_sum_l005_t095_v1"
OUTPUT_COLUMNS = (
    "model_version",
    "prediction_time",
    "odds_snapshot_time",
    "race_start_time",
    "odds_source",
    "race_id",
    "entry_id",
    "hrNo",
    "hrName",
    "winOdds_snapshot",
    "q_market",
    "p_premarket",
    "p_final",
    "market_delta",
    "break_even_prob",
    "expected_edge",
    "pred_rank",
    "action",
    "rejection_reason",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_global_request(frame: pd.DataFrame) -> None:
    required = {
        "race_id",
        "entry_id",
        "hrNo",
        "hrName",
        "dusu",
        "winOdds_snapshot",
        "prediction_time",
        "odds_snapshot_time",
        "race_start_time",
        "odds_source",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction request columns: {missing}")
    if frame["entry_id"].duplicated().any():
        raise ValueError("entry_id must be globally unique")
    if frame.empty:
        raise ValueError("Prediction request is empty")


def race_rejection_reasons(frame: pd.DataFrame) -> dict[str, str]:
    """Return one rejection reason per invalid race; absent races are valid."""
    validate_global_request(frame)
    reasons: dict[str, str] = {}
    for race_id, race in frame.groupby("race_id", sort=False):
        expected_counts = pd.to_numeric(race["dusu"], errors="coerce").dropna().unique()
        if len(expected_counts) != 1 or int(expected_counts[0]) != len(race):
            reasons[race_id] = "incomplete_race_entries"
            continue
        odds = pd.to_numeric(race["winOdds_snapshot"], errors="coerce")
        if odds.isna().any() or (odds <= 1.0).any():
            reasons[race_id] = "invalid_odds_snapshot"
            continue
        try:
            prediction_time = pd.to_datetime(race["prediction_time"], utc=True)
            snapshot_time = pd.to_datetime(race["odds_snapshot_time"], utc=True)
            start_time = pd.to_datetime(race["race_start_time"], utc=True)
        except (ValueError, TypeError):
            reasons[race_id] = "invalid_timestamp"
            continue
        if prediction_time.isna().any() or snapshot_time.isna().any() or start_time.isna().any():
            reasons[race_id] = "invalid_timestamp"
            continue
        if not (
            (snapshot_time <= prediction_time).all()
            and (prediction_time < start_time).all()
        ):
            reasons[race_id] = "invalid_time_order"
            continue
        for column in ("prediction_time", "odds_snapshot_time", "race_start_time"):
            if race[column].nunique(dropna=False) != 1:
                reasons[race_id] = f"inconsistent_{column}"
                break
    return reasons


def market_probability(frame: pd.DataFrame) -> np.ndarray:
    inverse_odds = 1.0 / frame["winOdds_snapshot"].to_numpy(dtype=float)
    totals = (
        pd.Series(inverse_odds)
        .groupby(frame["race_id"].to_numpy(), sort=False)
        .transform("sum")
        .to_numpy()
    )
    return inverse_odds / totals


def rejection_output(frame: pd.DataFrame, reasons: dict[str, str]) -> pd.DataFrame:
    rejected = frame.loc[frame["race_id"].isin(reasons)].copy()
    if rejected.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    output = rejected[
        [
            "prediction_time",
            "odds_snapshot_time",
            "race_start_time",
            "odds_source",
            "race_id",
            "entry_id",
            "hrNo",
            "hrName",
            "winOdds_snapshot",
        ]
    ].copy()
    output.insert(0, "model_version", MODEL_VERSION)
    for column in (
        "q_market",
        "p_premarket",
        "p_final",
        "market_delta",
        "break_even_prob",
        "expected_edge",
        "pred_rank",
    ):
        output[column] = np.nan
    output["action"] = "prediction_rejected"
    output["rejection_reason"] = output["race_id"].map(reasons)
    return output.loc[:, OUTPUT_COLUMNS]


def generate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    reasons = race_rejection_reasons(frame)
    rejected = rejection_output(frame, reasons)
    valid = frame.loc[~frame["race_id"].isin(reasons)].copy()
    if valid.empty:
        return rejected.sort_values(["race_id", "entry_id"], kind="stable").reset_index(
            drop=True
        )

    bundle = joblib.load(MODEL_ARTIFACT)
    raw = bundle["estimator"].predict_proba(
        model_frame(valid, bundle["feature_schema"])
    )[:, 1]
    normalization = load_json(NORMALIZATION_POLICY)["selected_methods"][MODEL_NAME]
    premarket = normalize_model_probabilities(valid, raw, method=normalization)
    q_market = market_probability(valid)
    lam = float(load_json(BLEND_POLICY)["selected_lambdas"][MODEL_NAME])
    blended = geometric_blend(valid, q_market, premarket, lam=lam)
    temperature = float(
        load_json(TEMPERATURE_POLICY)["selected_temperatures"][MODEL_NAME]
    )
    final = temperature_scale(valid, blended, temperature=temperature)
    betting = load_json(BETTING_POLICY)["deployment_policy"]

    output = valid[
        [
            "prediction_time",
            "odds_snapshot_time",
            "race_start_time",
            "odds_source",
            "race_id",
            "entry_id",
            "hrNo",
            "hrName",
            "winOdds_snapshot",
        ]
    ].copy()
    output.insert(0, "model_version", MODEL_VERSION)
    output["q_market"] = q_market
    output["p_premarket"] = premarket
    output["p_final"] = final
    output["market_delta"] = final - q_market
    output["break_even_prob"] = 1.0 / output["winOdds_snapshot"]
    output["expected_edge"] = final * output["winOdds_snapshot"] - 1.0
    output["pred_rank"] = (
        output.groupby("race_id", sort=False)["p_final"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    if betting["action"] == "bet":
        threshold = float(betting["threshold"])
        output["action"] = np.where(
            output["expected_edge"].ge(threshold), "bet", "no_bet"
        )
    else:
        output["action"] = "no_bet"
    output["rejection_reason"] = ""
    accepted = output.loc[:, OUTPUT_COLUMNS]
    combined = (
        pd.concat([accepted, rejected], ignore_index=True)
        if not rejected.empty
        else accepted.reset_index(drop=True)
    )
    combined = combined.sort_values(["race_id", "entry_id"], kind="stable").reset_index(
        drop=True
    )
    validate_prediction_output(combined)
    return combined


def validate_prediction_output(output: pd.DataFrame) -> None:
    if output.columns.tolist() != list(OUTPUT_COLUMNS):
        raise ValueError("Prediction output columns do not match the contract")
    if output["entry_id"].duplicated().any():
        raise ValueError("Prediction output entry_id must be unique")
    valid = output.loc[output["action"].ne("prediction_rejected")]
    if not valid["p_final"].between(0.0, 1.0).all():
        raise ValueError("p_final must be within [0, 1]")
    for column in ("q_market", "p_premarket", "p_final"):
        sums = valid.groupby("race_id")[column].sum().to_numpy()
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError(f"{column} must sum to one within valid races")
    if output.loc[output["action"].eq("prediction_rejected"), "rejection_reason"].eq("").any():
        raise ValueError("Rejected predictions require a reason")


def write_schema_manifest() -> dict:
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "model_artifact": {
            "path": "artifacts/models/m2_xgboost.joblib",
            "sha256": sha256_file(MODEL_ARTIFACT),
        },
        "columns": list(OUTPUT_COLUMNS),
        "required_input_metadata": [
            "prediction_time",
            "odds_snapshot_time",
            "race_start_time",
            "odds_source",
            "winOdds_snapshot",
        ],
        "contracts": {
            "valid_race_probability_sum": 1.0,
            "entry_id_unique": True,
            "result_columns_forbidden": ["win", "ord", "fin_rank", "fin_pct", "resid"],
            "incomplete_race_action": "prediction_rejected",
            "current_betting_action": "no_bet",
        },
        "policy_files": [
            "data/manifests/normalization_policy.json",
            "data/manifests/market_blend_policy.json",
            "data/manifests/temperature_policy.json",
            "data/manifests/betting_policy.json",
        ],
    }
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
