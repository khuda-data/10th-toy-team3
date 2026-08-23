"""Convert arbitrary race-ranking scores into valid probability simplexes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_scores_to_probabilities(
    frame: pd.DataFrame,
    scores,
    *,
    temperature: float,
) -> np.ndarray:
    """Return softmax(score / temperature) independently within each race."""
    if "race_id" not in frame:
        raise ValueError("race_id is required for ranking score softmax")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Ranking temperature must be finite and greater than zero")
    values = np.asarray(scores, dtype=float)
    if len(values) != len(frame):
        raise ValueError("Ranking score length does not match frame")
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores must be finite")

    scaled = values / float(temperature)
    work = pd.DataFrame(
        {"race_id": frame["race_id"].to_numpy(), "scaled_score": scaled}
    )
    maxima = (
        work.groupby("race_id", sort=False)["scaled_score"].transform("max").to_numpy()
    )
    numerators = np.exp(scaled - maxima)
    totals = (
        pd.Series(numerators)
        .groupby(frame["race_id"].to_numpy(), sort=False)
        .transform("sum")
        .to_numpy()
    )
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Ranking softmax produced an invalid race total")
    probabilities = numerators / totals
    sums = (
        pd.Series(probabilities)
        .groupby(frame["race_id"].to_numpy(), sort=False)
        .sum()
        .to_numpy()
    )
    if not np.allclose(sums, 1.0, atol=1e-9):
        raise ValueError("Ranking probabilities do not sum to one within race")
    return probabilities
