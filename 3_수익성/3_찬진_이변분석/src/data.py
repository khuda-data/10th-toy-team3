from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_ROWS,
    EXPECTED_SUBSETS,
    FINAL_CSV,
    FOLDS,
    MERGE_COLUMNS,
    SPLIT_ROOT,
    TARGETS,
)


def _read_final_columns(columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(FINAL_CSV, usecols=columns, low_memory=False)


def load_fold(fold: str, include_outcomes: bool = True) -> pd.DataFrame:
    if fold not in FOLDS:
        raise ValueError(f"Unknown fold: {fold}")
    split_path = SPLIT_ROOT / f"{fold}.csv"
    split = pd.read_csv(split_path, low_memory=False)
    if not split["entry_id"].is_unique:
        raise ValueError(f"{fold}: split entry_id is not unique")

    merge_cols = MERGE_COLUMNS if include_outcomes else ["entry_id", "pop_pct"]
    final = _read_final_columns(merge_cols)
    if not final["entry_id"].is_unique:
        raise ValueError("final.csv.gz: entry_id is not unique")

    overlap = (set(split.columns) & set(final.columns)) - {"entry_id"}
    if overlap:
        split = split.drop(columns=sorted(overlap))
    merged = split.merge(final, on="entry_id", how="left", validate="one_to_one")
    if len(merged) != len(split):
        raise ValueError(f"{fold}: row count changed after merge")
    return merged


def subset_and_target(df: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    spec = TARGETS[target_name]
    if spec["subset_op"] == "ge":
        mask = df[spec["subset_column"]].ge(spec["subset_value"])
    else:
        mask = df[spec["subset_column"]].le(spec["subset_value"])
    subset = df.loc[mask].copy().reset_index(drop=True)

    if target_name == "darkhorse":
        y = subset["place"].eq(1).astype("int8")
    else:
        y = subset["fin_pct"].ge(spec["target_value"]).astype("int8")
    return subset, y


def validate_stored_label(df: pd.DataFrame, target_name: str) -> float:
    subset, y = subset_and_target(df, target_name)
    stored = subset[TARGETS[target_name]["stored_label"]].astype("int8")
    return float(stored.eq(y).mean())


def preflight_report() -> dict:
    report: dict = {
        "files": {
            "final_csv": str(FINAL_CSV),
            "split_root": str(SPLIT_ROOT),
        },
        "folds": {},
        "label_validation": {},
        "test_outcomes_opened": False,
    }

    loaded: dict[str, pd.DataFrame] = {}
    for fold in ("train", "valid"):
        loaded[fold] = load_fold(fold, include_outcomes=True)
    loaded["test"] = load_fold("test", include_outcomes=False)

    seen_races: dict[str, set[str]] = {}
    for fold, df in loaded.items():
        row_count = len(df)
        if row_count != EXPECTED_ROWS[fold]:
            raise ValueError(f"{fold}: expected {EXPECTED_ROWS[fold]} rows, got {row_count}")
        seen_races[fold] = set(df["race_id"].astype(str))
        fold_info = {
            "rows": row_count,
            "entry_id_unique": bool(df["entry_id"].is_unique),
            "race_count": int(df["race_id"].nunique()),
            "date_min": int(pd.to_numeric(df["rcDate"], errors="coerce").min()),
            "date_max": int(pd.to_numeric(df["rcDate"], errors="coerce").max()),
            "pop_pct_missing": int(df["pop_pct"].isna().sum()),
            "subsets": {},
        }
        for target_name, expected in EXPECTED_SUBSETS.items():
            spec = TARGETS[target_name]
            if spec["subset_op"] == "ge":
                count = int(df[spec["subset_column"]].ge(spec["subset_value"]).sum())
            else:
                count = int(df[spec["subset_column"]].le(spec["subset_value"]).sum())
            if count != expected[fold]:
                raise ValueError(
                    f"{fold}/{target_name}: expected {expected[fold]}, got {count}"
                )
            fold_info["subsets"][target_name] = count
        report["folds"][fold] = fold_info

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = seen_races[left] & seen_races[right]
        if overlap:
            raise ValueError(f"race_id overlap between {left} and {right}: {len(overlap)}")

    for target_name in TARGETS:
        report["label_validation"][target_name] = {}
        for fold in ("train", "valid"):
            agreement = validate_stored_label(loaded[fold], target_name)
            if agreement != 1.0:
                raise ValueError(f"{fold}/{target_name}: stored-label agreement={agreement}")
            subset, y = subset_and_target(loaded[fold], target_name)
            report["label_validation"][target_name][fold] = {
                "agreement": agreement,
                "positives": int(y.sum()),
                "base_rate": float(y.mean()),
            }

    return report


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
