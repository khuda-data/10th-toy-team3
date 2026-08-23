"""Conservative market keep/switch gate features and adaptive blending."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


GATE_NUMERIC_FEATURES = (
    "q_market_top1",
    "market_top_gap",
    "market_entropy_normalized",
    "p_ranker_top1",
    "ranker_probability_gap",
    "ranker_entropy_normalized",
    "ranker_score_gap",
    "market_probability_of_ranker_top",
    "ranker_probability_of_market_top",
    "n_entries",
    "rcDist",
    "waterRate",
    "feature_missing_rate",
)
GATE_CATEGORICAL_FEATURES = ("rank", "track", "weather")
GATE_FEATURES = GATE_NUMERIC_FEATURES + GATE_CATEGORICAL_FEATURES


def _entropy(values: np.ndarray) -> float:
    positive = values[values > 0]
    return float(-(positive * np.log(positive)).sum())


def _ordered(race: pd.DataFrame, primary: str) -> pd.DataFrame:
    by = [primary]
    ascending = [False]
    if primary != "q_market":
        by.append("q_market")
        ascending.append(False)
    by.append("entry_id")
    ascending.append(True)
    return race.sort_values(by, ascending=ascending, kind="stable")


def build_gate_race_table(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Create one pre-race gate-feature row and an evaluation label per race."""
    required = {
        "race_id",
        "entry_id",
        "rcDate",
        "win",
        "q_market",
        "ranking_score",
        "p_ranker_race",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Gate input missing columns: {missing}")
    feature_columns = tuple(feature_columns)
    missing_features = sorted(set(feature_columns) - set(frame.columns))
    if missing_features:
        raise ValueError(f"Gate missingness columns unavailable: {missing_features}")
    if frame.empty or frame["entry_id"].duplicated().any():
        raise ValueError("Gate input must be non-empty with unique entry_id")
    if not frame.groupby("race_id", sort=False)["win"].sum().eq(1).all():
        raise ValueError("Gate input requires exactly one winner per race")
    for column in ("q_market", "p_ranker_race"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Gate probability {column} is invalid")
        sums = values.groupby(frame["race_id"], sort=False).sum().to_numpy()
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise ValueError(f"Gate probability {column} must sum to one by race")

    records: list[dict[str, object]] = []
    for race_id, race in frame.groupby("race_id", sort=False):
        market_order = _ordered(race, "q_market")
        ranker_order = _ordered(race, "ranking_score")
        market_top, market_second = market_order.iloc[0], market_order.iloc[1]
        ranker_top, ranker_second = ranker_order.iloc[0], ranker_order.iloc[1]
        winner = race.loc[race["win"].eq(1)].iloc[0]
        market_correct = bool(market_top["entry_id"] == winner["entry_id"])
        ranker_correct = bool(ranker_top["entry_id"] == winner["entry_id"])
        disagreement = bool(market_top["entry_id"] != ranker_top["entry_id"])
        n_entries = int(len(race))
        market_values = race["q_market"].to_numpy(dtype=float)
        ranker_values = race["p_ranker_race"].to_numpy(dtype=float)
        indexed = race.set_index("entry_id", drop=False)
        first = race.iloc[0]
        record = {
            "race_id": race_id,
            "rcDate": int(first["rcDate"]),
            "market_top_entry_id": market_top["entry_id"],
            "ranker_top_entry_id": ranker_top["entry_id"],
            "winner_entry_id": winner["entry_id"],
            "top1_disagreement": disagreement,
            "market_correct": market_correct,
            "ranker_correct": ranker_correct,
            "switch_beneficial": bool(disagreement and ranker_correct and not market_correct),
            "switch_harmful": bool(disagreement and market_correct and not ranker_correct),
            "both_wrong": bool(not market_correct and not ranker_correct),
            "q_market_top1": float(market_top["q_market"]),
            "market_top_gap": float(market_top["q_market"] - market_second["q_market"]),
            "market_entropy_normalized": float(_entropy(market_values) / np.log(n_entries)),
            "p_ranker_top1": float(ranker_top["p_ranker_race"]),
            "ranker_probability_gap": float(
                ranker_top["p_ranker_race"] - ranker_second["p_ranker_race"]
            ),
            "ranker_entropy_normalized": float(_entropy(ranker_values) / np.log(n_entries)),
            "ranker_score_gap": float(
                ranker_top["ranking_score"] - ranker_second["ranking_score"]
            ),
            "market_probability_of_ranker_top": float(
                indexed.loc[ranker_top["entry_id"], "q_market"]
            ),
            "ranker_probability_of_market_top": float(
                indexed.loc[market_top["entry_id"], "p_ranker_race"]
            ),
            "n_entries": n_entries,
            "rcDist": first["rcDist"] if "rcDist" in race else pd.NA,
            "waterRate": first["waterRate"] if "waterRate" in race else pd.NA,
            "rank": first["rank"] if "rank" in race else pd.NA,
            "track": first["track"] if "track" in race else pd.NA,
            "weather": first["weather"] if "weather" in race else pd.NA,
            "feature_missing_rate": float(
                race.loc[:, feature_columns].isna().to_numpy().mean()
            ),
        }
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values(
        ["rcDate", "race_id"], kind="stable"
    ).reset_index(drop=True)


def adaptive_geometric_blend(
    frame: pd.DataFrame,
    market_probabilities,
    ranker_probabilities,
    race_lambdas: pd.Series | dict[str, float],
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Apply a frozen lambda per race and return valid market-anchored probabilities."""
    market = np.asarray(market_probabilities, dtype=float)
    ranker = np.asarray(ranker_probabilities, dtype=float)
    if len(market) != len(frame) or len(ranker) != len(frame):
        raise ValueError("Adaptive blend probability length does not match frame")
    if not np.isfinite(market).all() or not np.isfinite(ranker).all():
        raise ValueError("Adaptive blend probabilities must be finite")
    if (market < 0).any() or (ranker < 0).any():
        raise ValueError("Adaptive blend probabilities must be non-negative")
    lambdas = pd.Series(race_lambdas, dtype=float)
    race_ids = pd.Index(frame["race_id"].astype(str).unique())
    missing = race_ids.difference(lambdas.index.astype(str))
    if len(missing):
        raise ValueError(f"Adaptive blend missing race lambdas: {missing.tolist()[:3]}")
    lambdas.index = lambdas.index.astype(str)
    row_lambda = frame["race_id"].astype(str).map(lambdas).to_numpy(dtype=float)
    if not np.isfinite(row_lambda).all() or (row_lambda < 0).any() or (row_lambda > 1).any():
        raise ValueError("Adaptive blend lambdas must be finite within [0, 1]")

    log_score = (1.0 - row_lambda) * np.log(np.clip(market, epsilon, None))
    log_score += row_lambda * np.log(np.clip(ranker, epsilon, None))
    work = pd.DataFrame(
        {"race_id": frame["race_id"].to_numpy(), "log_score": log_score}
    )
    maxima = work.groupby("race_id", sort=False)["log_score"].transform("max").to_numpy()
    numerator = np.exp(log_score - maxima)
    totals = (
        pd.Series(numerator)
        .groupby(frame["race_id"].to_numpy(), sort=False)
        .transform("sum")
        .to_numpy()
    )
    probabilities = numerator / totals
    sums = pd.Series(probabilities).groupby(frame["race_id"].to_numpy()).sum().to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-9):
        raise ValueError("Adaptive probabilities do not sum to one by race")
    return probabilities
