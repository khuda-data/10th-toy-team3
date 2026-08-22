"""Race-contiguous learning-to-rank dataset contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.registry import assert_feature_list, select_premarket_features


RANKING_SORT_KEYS = ("rcDate", "race_id", "entry_id")


@dataclass(frozen=True)
class RankingDataset:
    """Ordered entries and group metadata ready for an XGBoost ranker."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    relevance: np.ndarray
    group_sizes: np.ndarray
    group_ids: tuple[str, ...]

    @property
    def features(self) -> pd.DataFrame:
        return self.frame.loc[:, self.feature_names].copy()


def _assert_required_columns(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> None:
    required = {"race_id", "entry_id", "rcDate", "win", *feature_names}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing ranking columns: {missing}")


def _assert_race_integrity(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Ranking dataset cannot be empty")
    if frame["entry_id"].isna().any() or frame["entry_id"].duplicated().any():
        raise ValueError("Ranking entry_id values must be complete and unique")
    if frame[["race_id", "rcDate"]].isna().any().any():
        raise ValueError("Ranking race_id and rcDate must be complete")
    if not set(pd.Series(frame["win"]).dropna().unique()) <= {0, 1}:
        raise ValueError("Ranking relevance must be binary")

    grouped = frame.groupby("race_id", sort=False)
    if not grouped.size().ge(2).all():
        raise ValueError("Every ranking group must contain at least two entries")
    if not grouped["win"].sum().eq(1).all():
        raise ValueError("Every ranking group must contain exactly one winner")
    if not grouped["rcDate"].nunique().eq(1).all():
        raise ValueError("A race_id cannot span multiple dates")


def _assert_group_contiguity(
    ordered: pd.DataFrame, group_ids: tuple[str, ...], group_sizes: np.ndarray
) -> None:
    expanded = np.repeat(np.asarray(group_ids, dtype=object), group_sizes)
    actual = ordered["race_id"].astype(str).to_numpy(dtype=object)
    if len(expanded) != len(actual) or not np.array_equal(expanded, actual):
        raise ValueError("race_id rows must be contiguous and match group_sizes")
    if int(group_sizes.sum()) != len(ordered):
        raise ValueError("group_sizes must sum to the number of entry rows")
    dates = ordered["rcDate"].to_numpy()
    if len(dates) > 1 and np.any(dates[1:] < dates[:-1]):
        raise ValueError("Ranking groups must be chronological")


def build_ranking_dataset(
    frame: pd.DataFrame,
    *,
    feature_names: Iterable[str] | None = None,
) -> RankingDataset:
    """Sort complete races and create binary relevance plus XGBoost group sizes."""
    selected = tuple(
        select_premarket_features() if feature_names is None else feature_names
    )
    if not selected:
        raise ValueError("At least one ranking feature is required")
    if len(selected) != len(set(selected)):
        raise ValueError("Ranking feature names must be unique")
    assert_feature_list(selected, model_kind="ranking")
    _assert_required_columns(frame, selected)
    _assert_race_integrity(frame)

    ordered = frame.sort_values(list(RANKING_SORT_KEYS), kind="stable").reset_index(
        drop=True
    )
    sizes = ordered.groupby("race_id", sort=False).size()
    group_ids = tuple(str(value) for value in sizes.index)
    group_sizes = sizes.to_numpy(dtype=np.uint32)
    relevance = ordered["win"].to_numpy(dtype=np.uint8)
    _assert_group_contiguity(ordered, group_ids, group_sizes)

    offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(group_sizes, dtype=np.int64))
    )
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        if int(relevance[start:stop].sum()) != 1:
            raise ValueError("Relevance vector is misaligned with ranking groups")

    return RankingDataset(
        frame=ordered,
        feature_names=selected,
        relevance=relevance,
        group_sizes=group_sizes,
        group_ids=group_ids,
    )


def build_ranking_manifests(
    dataset: RankingDataset,
    *,
    model_fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create auditable entry/group manifests without persisting feature values."""
    frame = dataset.frame
    group_sizes = dataset.group_sizes.astype(int)
    cumulative_sizes = np.cumsum(group_sizes, dtype=np.int64)
    row_starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), cumulative_sizes[:-1])
    )
    row_stops = cumulative_sizes
    group_positions = np.repeat(np.arange(len(group_sizes)), group_sizes)
    positions_in_group = np.concatenate(
        [np.arange(size, dtype=int) for size in group_sizes]
    )
    repeated_sizes = np.repeat(group_sizes, group_sizes)
    missing_counts = frame.loc[:, dataset.feature_names].isna().sum(axis=1).astype(int)

    entries = frame[["race_id", "entry_id", "rcDate"]].copy()
    entries.insert(0, "model_fold", model_fold)
    entries.insert(1, "row_position", np.arange(len(frame), dtype=int))
    entries["group_position"] = group_positions
    entries["position_in_group"] = positions_in_group
    entries["group_size"] = repeated_sizes
    entries["relevance"] = dataset.relevance
    entries["feature_missing_count"] = missing_counts.to_numpy()

    group_records: list[dict[str, object]] = []
    for position, (race_id, start, stop, size) in enumerate(
        zip(
            dataset.group_ids,
            row_starts,
            row_stops,
            group_sizes,
            strict=True,
        )
    ):
        group = frame.iloc[int(start) : int(stop)]
        winner = group.loc[group["win"].eq(1)].iloc[0]
        group_records.append(
            {
                "model_fold": model_fold,
                "group_position": position,
                "race_id": race_id,
                "rcDate": int(group["rcDate"].iloc[0]),
                "row_start": int(start),
                "row_stop_exclusive": int(stop),
                "group_size": int(size),
                "relevance_sum": int(group["win"].sum()),
                "winner_entry_id": winner["entry_id"],
                "winner_position_in_group": int(
                    group.index.get_loc(winner.name)
                    if winner.name in group.index
                    else np.flatnonzero(group["entry_id"].eq(winner["entry_id"]))[0]
                ),
            }
        )
    groups = pd.DataFrame.from_records(group_records)
    return entries, groups


def validate_ranking_manifests(
    entries: pd.DataFrame,
    groups: pd.DataFrame,
) -> None:
    """Fail closed when persisted row/group boundaries no longer align."""
    if entries.empty or groups.empty:
        raise ValueError("Ranking manifests cannot be empty")
    if entries["entry_id"].duplicated().any() or groups["race_id"].duplicated().any():
        raise ValueError("Ranking manifests require unique entry_id and race_id")
    if not groups["relevance_sum"].eq(1).all():
        raise ValueError("Every persisted ranking group must have relevance_sum=1")
    if not groups["group_size"].ge(2).all():
        raise ValueError("Every persisted ranking group must contain at least two entries")

    for model_fold, fold_groups in groups.groupby("model_fold", sort=False):
        fold_entries = entries.loc[entries["model_fold"].eq(model_fold)]
        ordered_groups = fold_groups.sort_values("group_position", kind="stable")
        if not np.array_equal(
            ordered_groups["group_position"].to_numpy(),
            np.arange(len(ordered_groups)),
        ):
            raise ValueError("group_position must be zero-based and contiguous per fold")
        if not np.array_equal(
            fold_entries.sort_values("row_position")["row_position"].to_numpy(),
            np.arange(len(fold_entries)),
        ):
            raise ValueError("row_position must be zero-based and contiguous per fold")
        if int(ordered_groups["group_size"].sum()) != len(fold_entries):
            raise ValueError("Persisted group sizes do not sum to fold rows")
        for row in ordered_groups.itertuples(index=False):
            members = fold_entries.loc[
                (fold_entries["row_position"] >= row.row_start)
                & (fold_entries["row_position"] < row.row_stop_exclusive)
            ]
            if len(members) != row.group_size or members["race_id"].nunique() != 1:
                raise ValueError("Persisted group row interval is invalid")
            if str(members["race_id"].iloc[0]) != str(row.race_id):
                raise ValueError("Persisted group interval points to another race")
            if int(members["relevance"].sum()) != 1:
                raise ValueError("Persisted relevance is invalid")
