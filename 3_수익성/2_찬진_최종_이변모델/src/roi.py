from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PERCENTILES
from .metrics import top_mask


def cluster_bootstrap_roi(
    selected: pd.DataFrame,
    reps: int = 5_000,
    seed: int = 42,
) -> tuple[float, float]:
    if selected.empty:
        return float("nan"), float("nan")
    grouped = selected.groupby("race_id", sort=False)["realized_return"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    race_n = len(grouped)
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=float)
    chunk = 250
    probabilities = np.full(race_n, 1.0 / race_n)
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        weights = rng.multinomial(race_n, probabilities, size=size)
        values[start:start + size] = (weights @ sums) / (weights @ counts)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _summarize(selected: pd.DataFrame, base_rate: float, label: str) -> dict:
    if selected.empty:
        return {"range": label, "bets": 0}
    hit_rate = float(selected["target"].mean())
    realized_roi = float(selected["realized_return"].mean())
    low, high = cluster_bootstrap_roi(selected)
    remove_one = selected.drop(index=selected["realized_return"].idxmax())
    return {
        "range": label,
        "bets": int(len(selected)),
        "hits": int(selected["target"].sum()),
        "hit_rate": hit_rate,
        "lift": float(hit_rate / base_rate),
        "mean_plc_odds": float(selected["plcOdds"].mean()),
        "median_plc_odds": float(selected["plcOdds"].median()),
        "expected_roi": float(selected["expected_return"].mean()),
        "realized_roi": realized_roi,
        "roi_ci_low": low,
        "roi_ci_high": high,
        "roi_without_top1": float(remove_one["realized_return"].mean()) if len(remove_one) else float("nan"),
    }


def percentile_roi_tables(
    frame: pd.DataFrame,
    score_column: str = "score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"race_id", "target", "plcOdds", "calibrated_probability", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ROI columns missing: {sorted(missing)}")
    data = frame.copy()
    data["realized_return"] = data["target"] * data["plcOdds"] - 1.0
    data["expected_return"] = data["calibrated_probability"] * data["plcOdds"] - 1.0
    base_rate = float(data["target"].mean())

    cumulative: list[dict] = []
    previous = np.zeros(len(data), dtype=bool)
    bands: list[dict] = []
    previous_fraction = 0.0
    scores = data[score_column].to_numpy(float)
    for fraction in PERCENTILES:
        current = top_mask(scores, fraction)
        cumulative.append(_summarize(data.loc[current], base_rate, f"top_{int(fraction * 100)}pct"))
        band = current & ~previous
        bands.append(
            _summarize(
                data.loc[band],
                base_rate,
                f"{int(previous_fraction * 100)}_{int(fraction * 100)}pct",
            )
        )
        previous = current
        previous_fraction = fraction
    return pd.DataFrame(cumulative), pd.DataFrame(bands)
