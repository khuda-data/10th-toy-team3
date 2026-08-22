"""Deterministic metrics for one-winner-per-race ranking predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_entries(
    frame: pd.DataFrame,
    scores,
    *,
    score_name: str = "ranking_score",
) -> pd.DataFrame:
    """Assign one unique position per race using the frozen tie-break contract."""
    required = {"race_id", "entry_id", "win", "q_market"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing ranking metric columns: {missing}")
    if frame["entry_id"].duplicated().any():
        raise ValueError("Ranking metrics require unique entry_id values")
    values = np.asarray(scores, dtype=float)
    if len(values) != len(frame):
        raise ValueError("Ranking score length does not match frame")
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores must be finite")
    winners = frame.groupby("race_id", sort=False)["win"].sum()
    if not winners.eq(1).all():
        raise ValueError("Ranking metrics require exactly one winner per race")

    work = frame[["race_id", "entry_id", "win", "q_market"]].copy()
    work[score_name] = values
    work = work.sort_values(
        ["race_id", score_name, "q_market", "entry_id"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    work["rank_position"] = work.groupby("race_id", sort=False).cumcount() + 1
    return work.sort_index(kind="stable")


def ranking_metrics(frame: pd.DataFrame, scores) -> dict[str, float | int]:
    """Evaluate Top-1, Hit@3, MRR, winner rank, and score separation."""
    ranked = rank_entries(frame, scores)
    winner_ranks = ranked.loc[ranked["win"].eq(1), "rank_position"].to_numpy(
        dtype=float
    )
    ordered = ranked.sort_values(
        ["race_id", "rank_position"], kind="stable"
    )
    first_two = ordered.groupby("race_id", sort=False)["ranking_score"].head(2)
    margins = first_two.groupby(ordered.loc[first_two.index, "race_id"], sort=False).agg(
        lambda values: float(values.iloc[0] - values.iloc[1])
    )
    return {
        "rows": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "top1_correct": int(np.sum(winner_ranks == 1)),
        "top1_accuracy": float(np.mean(winner_ranks == 1)),
        "hit_at_3": float(np.mean(winner_ranks <= 3)),
        "ndcg_at_1": float(np.mean(winner_ranks == 1)),
        "ndcg_at_3": float(
            np.mean(
                np.where(
                    winner_ranks <= 3,
                    1.0 / np.log2(winner_ranks + 1.0),
                    0.0,
                )
            )
        ),
        "mean_reciprocal_rank": float(np.mean(1.0 / winner_ranks)),
        "winner_mean_rank": float(np.mean(winner_ranks)),
        "winner_median_rank": float(np.median(winner_ranks)),
        "top_score_margin_mean": float(margins.mean()),
        "top_score_margin_median": float(margins.median()),
    }


def compare_ranking_metrics(
    market: dict[str, float | int],
    m2: dict[str, float | int],
    ranker: dict[str, float | int],
) -> dict[str, object]:
    """Return ranker deltas with consistent positive-is-better directions."""
    return {
        "r2_minus_market": {
            "top1_correct": int(ranker["top1_correct"] - market["top1_correct"]),
            "top1_accuracy": float(ranker["top1_accuracy"] - market["top1_accuracy"]),
            "hit_at_3": float(ranker["hit_at_3"] - market["hit_at_3"]),
            "mean_reciprocal_rank": float(
                ranker["mean_reciprocal_rank"] - market["mean_reciprocal_rank"]
            ),
            "winner_mean_rank_improvement": float(
                market["winner_mean_rank"] - ranker["winner_mean_rank"]
            ),
        },
        "r2_minus_m2": {
            "top1_correct": int(ranker["top1_correct"] - m2["top1_correct"]),
            "top1_accuracy": float(ranker["top1_accuracy"] - m2["top1_accuracy"]),
            "hit_at_3": float(ranker["hit_at_3"] - m2["hit_at_3"]),
            "mean_reciprocal_rank": float(
                ranker["mean_reciprocal_rank"] - m2["mean_reciprocal_rank"]
            ),
            "winner_mean_rank_improvement": float(
                m2["winner_mean_rank"] - ranker["winner_mean_rank"]
            ),
        },
    }
