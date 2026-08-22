"""Load race-level model folds by joining entries to the split manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.load_interim import load_seoul_interim
from src.data.load_raw import PROJECT_ROOT


DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "interim" / "split_manifest.csv"
KNOWN_FOLDS = {"train", "calibration", "test", "excluded"}


def load_model_entries(
    folds: Iterable[str] = ("train",),
    *,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
) -> pd.DataFrame:
    """Return complete races from requested folds, preserving chronological order."""
    requested = tuple(folds)
    unknown = set(requested) - KNOWN_FOLDS
    if unknown:
        raise ValueError(f"Unknown model folds: {sorted(unknown)}")

    entries = load_seoul_interim()
    split = pd.read_csv(split_path, encoding="utf-8-sig")
    if split["race_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate race_id values")

    merged = entries.merge(
        split[["race_id", "model_fold"]],
        on="race_id",
        how="left",
        validate="many_to_one",
    )
    if merged["model_fold"].isna().any():
        raise ValueError("Some entries are missing from the split manifest")

    selected = merged.loc[merged["model_fold"].isin(requested)].copy()
    selected = selected.sort_values(
        ["rcDate", "race_id", "entry_id"], kind="stable"
    ).reset_index(drop=True)
    return selected


def make_walk_forward_folds(train: pd.DataFrame) -> list[dict[str, object]]:
    """Create four expanding date-aligned 60/10 through 90/10 folds."""
    dates = sorted(train["rcDate"].unique().tolist())
    folds: list[dict[str, object]] = []
    for index, train_fraction in enumerate((0.6, 0.7, 0.8, 0.9), start=1):
        train_stop = max(1, int(len(dates) * train_fraction))
        valid_stop = len(dates) if index == 4 else int(len(dates) * (train_fraction + 0.1))
        train_dates = set(dates[:train_stop])
        valid_dates = set(dates[train_stop:valid_stop])
        train_mask = train["rcDate"].isin(train_dates)
        valid_mask = train["rcDate"].isin(valid_dates)
        if not valid_mask.any():
            raise ValueError(f"Walk-forward fold {index} has no validation rows")
        folds.append(
            {
                "fold": index,
                "train_index": train.index[train_mask].to_numpy(),
                "valid_index": train.index[valid_mask].to_numpy(),
                "train_date_max": int(max(train_dates)),
                "valid_date_min": int(min(valid_dates)),
                "valid_date_max": int(max(valid_dates)),
            }
        )
    return folds
