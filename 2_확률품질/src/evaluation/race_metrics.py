"""Race-normalized probabilities and primary evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-15
NORMALIZATION_METHODS = ("sum", "logit_softmax")


def normalize_by_race(frame: pd.DataFrame, raw_scores) -> np.ndarray:
    """Backward-compatible alias for sum normalization."""
    return normalize_model_probabilities(frame, raw_scores, method="sum")


def normalize_model_probabilities(
    frame: pd.DataFrame,
    raw_scores,
    *,
    method: str,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Convert entry-level binary probabilities to a race probability simplex."""
    if method not in NORMALIZATION_METHODS:
        raise ValueError(
            f"Unknown normalization method: {method}; expected {NORMALIZATION_METHODS}"
        )
    scores = np.asarray(raw_scores, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("Score length does not match frame length")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Raw scores must be finite and non-negative")
    if method == "logit_softmax" and (scores > 1).any():
        raise ValueError("logit_softmax requires raw probabilities in [0, 1]")

    if method == "sum":
        race_scores = scores
    else:
        clipped = np.clip(scores, epsilon, 1.0 - epsilon)
        logits = np.log(clipped) - np.log1p(-clipped)
        work_logits = pd.DataFrame(
            {"race_id": frame["race_id"].to_numpy(), "logit": logits}
        )
        maxima = (
            work_logits.groupby("race_id", sort=False)["logit"]
            .transform("max")
            .to_numpy()
        )
        race_scores = np.exp(logits - maxima)

    work = pd.DataFrame(
        {"race_id": frame["race_id"].to_numpy(), "score": race_scores}
    )
    totals = (
        work.groupby("race_id", sort=False)["score"].transform("sum").to_numpy()
    )
    if (totals <= 0).any():
        raise ValueError("Every race must have positive total score")
    return race_scores / totals


def geometric_blend(
    frame: pd.DataFrame,
    market_probabilities,
    model_probabilities,
    *,
    lam: float,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Geometrically blend market/model probabilities and normalize per race."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be within [0, 1]")
    market = np.asarray(market_probabilities, dtype=float)
    model = np.asarray(model_probabilities, dtype=float)
    if len(market) != len(frame) or len(model) != len(frame):
        raise ValueError("Probability length does not match frame length")
    if (
        not np.isfinite(market).all()
        or not np.isfinite(model).all()
        or (market < 0).any()
        or (model < 0).any()
    ):
        raise ValueError("Blend inputs must be finite and non-negative")

    log_score = (1.0 - lam) * np.log(np.clip(market, epsilon, None))
    log_score += lam * np.log(np.clip(model, epsilon, None))
    work = pd.DataFrame(
        {"race_id": frame["race_id"].to_numpy(), "log_score": log_score}
    )
    maxima = (
        work.groupby("race_id", sort=False)["log_score"]
        .transform("max")
        .to_numpy()
    )
    numerator = np.exp(log_score - maxima)
    totals = (
        pd.Series(numerator)
        .groupby(frame["race_id"].to_numpy(), sort=False)
        .transform("sum")
        .to_numpy()
    )
    return numerator / totals


def temperature_scale(
    frame: pd.DataFrame,
    probabilities,
    *,
    temperature: float,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Apply p_i(T) = softmax(log(p_i) / T) within every race."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    values = np.asarray(probabilities, dtype=float)
    if len(values) != len(frame):
        raise ValueError("Probability length does not match frame length")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Temperature input must be finite and non-negative")

    scores = np.log(np.clip(values, epsilon, None)) / temperature
    work = pd.DataFrame(
        {"race_id": frame["race_id"].to_numpy(), "score": scores}
    )
    maxima = (
        work.groupby("race_id", sort=False)["score"]
        .transform("max")
        .to_numpy()
    )
    numerator = np.exp(scores - maxima)
    totals = (
        pd.Series(numerator)
        .groupby(frame["race_id"].to_numpy(), sort=False)
        .transform("sum")
        .to_numpy()
    )
    return numerator / totals


def race_metrics(frame: pd.DataFrame, probabilities) -> dict[str, float | int]:
    p = np.asarray(probabilities, dtype=float)
    y = frame["win"].to_numpy(dtype=int)
    if len(p) != len(frame):
        raise ValueError("Probability length does not match frame length")
    sums = pd.Series(p).groupby(frame["race_id"].to_numpy()).sum().to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-9):
        raise ValueError("Probabilities must sum to one within each race")

    work = pd.DataFrame(
        {"race_id": frame["race_id"].to_numpy(), "y": y, "p": p}
    )
    winner_rows = work.groupby("race_id", sort=False)["y"].sum()
    if not winner_rows.eq(1).all():
        raise ValueError("Metrics require exactly one winner per race")

    winner_p = work.loc[work["y"].eq(1), "p"].clip(EPSILON, 1.0)
    race_log_loss = float(-np.log(winner_p).mean())
    race_brier = float(
        work.assign(sq=(work["p"] - work["y"]) ** 2)
        .groupby("race_id", sort=False)["sq"]
        .sum()
        .mean()
    )
    work["rank"] = work.groupby("race_id", sort=False)["p"].rank(
        method="min", ascending=False
    )
    winner_rank = work.loc[work["y"].eq(1), "rank"]
    return {
        "rows": int(len(work)),
        "races": int(work["race_id"].nunique()),
        "race_log_loss": race_log_loss,
        "race_brier": race_brier,
        "top1_accuracy": float(winner_rank.eq(1).mean()),
        "mean_reciprocal_rank": float((1.0 / winner_rank).mean()),
        "probability_sum_max_abs_error": float(np.max(np.abs(sums - 1.0))),
    }
