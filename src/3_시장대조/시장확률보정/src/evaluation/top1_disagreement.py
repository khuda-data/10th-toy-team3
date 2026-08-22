"""Deterministic race-level Top-1 disagreement analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


PROBABILITY_TOLERANCE = 1e-6
CORRECTNESS_CASES = (
    "both_correct",
    "market_only_correct",
    "model_only_correct",
    "both_wrong",
)


def _validate_entries(frame: pd.DataFrame, model_probability_col: str) -> None:
    required = {
        "race_id",
        "entry_id",
        "rcDate",
        "win",
        "q_market",
        model_probability_col,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing disagreement columns: {missing}")
    if frame.empty:
        raise ValueError("Disagreement analysis requires at least one race")
    if frame["entry_id"].duplicated().any():
        raise ValueError("entry_id must be unique")
    if frame[["race_id", "entry_id", "rcDate"]].isna().any().any():
        raise ValueError("Race identifiers and dates must be complete")

    winners = frame.groupby("race_id", sort=False)["win"].sum()
    if not winners.eq(1).all():
        raise ValueError("Every analyzed race must have exactly one winner")
    sizes = frame.groupby("race_id", sort=False).size()
    if not sizes.ge(2).all():
        raise ValueError("Every analyzed race must have at least two entries")

    for column in ("q_market", model_probability_col):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite non-negative probabilities")
        sums = values.groupby(frame["race_id"], sort=False).sum()
        if not np.allclose(sums.to_numpy(), 1.0, atol=PROBABILITY_TOLERANCE):
            raise ValueError(f"{column} must sum to one within each race")


def _ordered_entries(
    race: pd.DataFrame,
    *,
    primary_col: str,
    market_tie_break: bool,
) -> pd.DataFrame:
    by = [primary_col]
    ascending = [False]
    if market_tie_break and primary_col != "q_market":
        by.append("q_market")
        ascending.append(False)
    by.append("entry_id")
    ascending.append(True)
    return race.sort_values(by, ascending=ascending, kind="stable")


def _entropy(probabilities: pd.Series) -> float:
    values = probabilities.to_numpy(dtype=float)
    positive = values[values > 0]
    return float(-(positive * np.log(positive)).sum())


def build_race_disagreement_table(
    frame: pd.DataFrame,
    *,
    source: str,
    model_probability_col: str = "p_model_race",
    feature_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Collapse entry predictions into one deterministic row per race."""
    _validate_entries(frame, model_probability_col)
    feature_columns = tuple(feature_columns)
    unknown_features = sorted(set(feature_columns) - set(frame.columns))
    if unknown_features:
        raise ValueError(f"Unknown missingness features: {unknown_features}")

    condition_columns = (
        "rcDist",
        "rank",
        "track",
        "weather",
        "waterRate",
        "n_run",
    )
    records: list[dict[str, object]] = []
    for race_id, race in frame.groupby("race_id", sort=False):
        market_order = _ordered_entries(
            race, primary_col="q_market", market_tie_break=False
        )
        model_order = _ordered_entries(
            race, primary_col=model_probability_col, market_tie_break=True
        )
        market_top = market_order.iloc[0]
        model_top = model_order.iloc[0]
        winner = race.loc[race["win"].eq(1)].iloc[0]
        market_correct = bool(market_top["entry_id"] == winner["entry_id"])
        model_correct = bool(model_top["entry_id"] == winner["entry_id"])
        if market_correct and model_correct:
            correctness_case = "both_correct"
        elif market_correct:
            correctness_case = "market_only_correct"
        elif model_correct:
            correctness_case = "model_only_correct"
        else:
            correctness_case = "both_wrong"

        n_entries = int(len(race))
        market_entropy = _entropy(race["q_market"])
        model_entropy = _entropy(race[model_probability_col])
        market_top2 = market_order.iloc[1]
        model_top2 = model_order.iloc[1]
        market_entry = race.set_index("entry_id", drop=False).loc[market_top["entry_id"]]
        model_entry = race.set_index("entry_id", drop=False).loc[model_top["entry_id"]]
        record: dict[str, object] = {
            "source": source,
            "race_id": race_id,
            "rcDate": int(race["rcDate"].iloc[0]),
            "wf_fold": (
                int(race["wf_fold"].iloc[0])
                if "wf_fold" in race and pd.notna(race["wf_fold"].iloc[0])
                else pd.NA
            ),
            "n_entries": n_entries,
            "winner_entry_id": winner["entry_id"],
            "market_top_entry_id": market_top["entry_id"],
            "model_top_entry_id": model_top["entry_id"],
            "market_correct": market_correct,
            "model_correct": model_correct,
            "correctness_case": correctness_case,
            "top1_disagreement": bool(
                market_top["entry_id"] != model_top["entry_id"]
            ),
            "q_market_top1": float(market_top["q_market"]),
            "q_market_top2": float(market_top2["q_market"]),
            "market_top_gap": float(
                market_top["q_market"] - market_top2["q_market"]
            ),
            "market_entropy": market_entropy,
            "market_entropy_normalized": float(market_entropy / np.log(n_entries)),
            "p_model_top1": float(model_top[model_probability_col]),
            "p_model_top2": float(model_top2[model_probability_col]),
            "model_top_gap": float(
                model_top[model_probability_col] - model_top2[model_probability_col]
            ),
            "model_entropy": model_entropy,
            "model_entropy_normalized": float(model_entropy / np.log(n_entries)),
            "model_probability_of_market_top": float(
                market_entry[model_probability_col]
            ),
            "market_probability_of_model_top": float(model_entry["q_market"]),
            "winner_q_market": float(winner["q_market"]),
            "winner_p_model": float(winner[model_probability_col]),
            "feature_missing_rate": (
                float(race.loc[:, feature_columns].isna().to_numpy().mean())
                if feature_columns
                else 0.0
            ),
        }
        for column in condition_columns:
            record[column] = race[column].iloc[0] if column in race else pd.NA
        records.append(record)

    output = pd.DataFrame.from_records(records)
    if set(output["correctness_case"].unique()) - set(CORRECTNESS_CASES):
        raise AssertionError("Unexpected correctness case")
    return output.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)


def add_segment_bands(races: pd.DataFrame) -> pd.DataFrame:
    """Add predeclared, interpretable bands used by the stage-22 report."""
    output = races.copy()
    output["market_gap_band"] = pd.cut(
        output["market_top_gap"],
        bins=[-np.inf, 0.02, 0.05, 0.10, np.inf],
        labels=["<=0.02", "0.02-0.05", "0.05-0.10", ">0.10"],
        right=True,
    ).astype("string")
    output["model_gap_band"] = pd.cut(
        output["model_top_gap"],
        bins=[-np.inf, 0.02, 0.05, 0.10, np.inf],
        labels=["<=0.02", "0.02-0.05", "0.05-0.10", ">0.10"],
        right=True,
    ).astype("string")
    output["market_entropy_band"] = pd.cut(
        output["market_entropy_normalized"],
        bins=[-np.inf, 0.70, 0.85, np.inf],
        labels=["<0.70", "0.70-0.85", ">0.85"],
        right=False,
    ).astype("string")
    output["field_size_band"] = pd.cut(
        output["n_entries"],
        bins=[-np.inf, 8, 11, np.inf],
        labels=["<=8", "9-11", ">=12"],
        right=True,
    ).astype("string")
    output["missing_rate_band"] = pd.cut(
        output["feature_missing_rate"],
        bins=[-np.inf, 0, 0.05, np.inf],
        labels=["0", "0-0.05", ">0.05"],
        right=True,
    ).astype("string")
    return output


def _summary_record(
    group: pd.DataFrame,
    *,
    source: str,
    segment_name: str,
    segment_value: str,
) -> dict[str, object]:
    disagreement = group.loc[group["top1_disagreement"]]
    cases = group["correctness_case"].value_counts()
    disagreement_cases = disagreement["correctness_case"].value_counts()
    races = int(len(group))
    disagreement_races = int(len(disagreement))
    return {
        "source": source,
        "segment_name": segment_name,
        "segment_value": segment_value,
        "races": races,
        "disagreement_races": disagreement_races,
        "disagreement_rate": disagreement_races / races,
        "market_correct": int(group["market_correct"].sum()),
        "model_correct": int(group["model_correct"].sum()),
        "market_top1_accuracy": float(group["market_correct"].mean()),
        "model_top1_accuracy": float(group["model_correct"].mean()),
        "model_minus_market_correct": int(
            group["model_correct"].sum() - group["market_correct"].sum()
        ),
        "delta_top1": float(
            group["model_correct"].mean() - group["market_correct"].mean()
        ),
        "both_correct": int(cases.get("both_correct", 0)),
        "market_only_correct": int(cases.get("market_only_correct", 0)),
        "model_only_correct": int(cases.get("model_only_correct", 0)),
        "both_wrong": int(cases.get("both_wrong", 0)),
        "disagreement_market_only_correct": int(
            disagreement_cases.get("market_only_correct", 0)
        ),
        "disagreement_model_only_correct": int(
            disagreement_cases.get("model_only_correct", 0)
        ),
        "disagreement_both_wrong": int(disagreement_cases.get("both_wrong", 0)),
        "model_override_win_rate": (
            float(disagreement["model_correct"].mean())
            if disagreement_races
            else np.nan
        ),
    }


def build_segment_summary(races: pd.DataFrame) -> pd.DataFrame:
    """Summarize correctness and disagreement outcomes across fixed segments."""
    work = add_segment_bands(races)
    segment_columns = (
        "market_gap_band",
        "model_gap_band",
        "market_entropy_band",
        "field_size_band",
        "missing_rate_band",
        "rcDist",
        "rank",
        "track",
        "weather",
    )
    records: list[dict[str, object]] = []
    source_groups = [("combined", work), *work.groupby("source", sort=False)]
    for source, source_frame in source_groups:
        records.append(
            _summary_record(
                source_frame,
                source=str(source),
                segment_name="all",
                segment_value="all",
            )
        )
        for column in segment_columns:
            values = source_frame[column].astype("string").fillna("<missing>")
            for value, indices in values.groupby(values, sort=True).groups.items():
                records.append(
                    _summary_record(
                        source_frame.loc[indices],
                        source=str(source),
                        segment_name=column,
                        segment_value=str(value),
                    )
                )
    return pd.DataFrame.from_records(records)
