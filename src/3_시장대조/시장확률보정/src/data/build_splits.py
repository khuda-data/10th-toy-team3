"""Build the race-level chronological split manifest for Seoul modeling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_interim import DEFAULT_SEOUL_INTERIM_PATH, load_seoul_interim
from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "split_manifest.csv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.json"

TRAIN_END = 20_250_511
CALIBRATION_START = 20_250_517
CALIBRATION_END = 20_251_227
TEST_START = 20_251_228

EXPECTED_FOLDS = {
    "train": {"rows": 19_617, "races": 1_891},
    "calibration": {"rows": 6_582, "races": 641},
    "test": {"rows": 6_639, "races": 635},
    "excluded": {"rows": 50, "races": 5},
}


def build_split_frame(entries: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    races = (
        entries.groupby("race_id", as_index=False)
        .agg(
            rcDate=("rcDate", "first"),
            entry_count=("entry_id", "size"),
            winner_rows=("win", "sum"),
            race_status=("race_status", "first"),
            eligible_primary=("eligible_primary", "first"),
        )
        .sort_values(["rcDate", "race_id"], kind="stable")
        .reset_index(drop=True)
    )

    eligible = races["eligible_primary"].astype(bool)
    conditions = [
        ~eligible,
        eligible & races["rcDate"].le(TRAIN_END),
        eligible
        & races["rcDate"].ge(CALIBRATION_START)
        & races["rcDate"].le(CALIBRATION_END),
        eligible & races["rcDate"].ge(TEST_START),
    ]
    races["model_fold"] = np.select(
        conditions,
        ["excluded", "train", "calibration", "test"],
        default="unassigned",
    )
    races["exclusion_reason"] = np.where(
        races["model_fold"].eq("excluded"),
        races["race_status"],
        "",
    )

    if races["model_fold"].eq("unassigned").any():
        bad = races.loc[races["model_fold"].eq("unassigned"), ["race_id", "rcDate"]]
        raise ValueError(f"Eligible races fall outside configured date windows:\n{bad}")
    if races["race_id"].duplicated().any():
        raise ValueError("race_id is duplicated in split manifest")

    entry_fold = entries[["race_id", "entry_id"]].merge(
        races[["race_id", "model_fold"]],
        on="race_id",
        how="left",
        validate="many_to_one",
    )
    fold_summary: dict[str, Any] = {}
    for fold, expected in EXPECTED_FOLDS.items():
        race_part = races.loc[races["model_fold"].eq(fold)]
        entry_part = entry_fold.loc[entry_fold["model_fold"].eq(fold)]
        actual = {"rows": int(len(entry_part)), "races": int(len(race_part))}
        if actual != expected:
            raise ValueError(f"Unexpected {fold} counts: {actual} != {expected}")
        fold_summary[fold] = {
            **actual,
            "date_min": int(race_part["rcDate"].min()),
            "date_max": int(race_part["rcDate"].max()),
        }

    dates_by_fold = {
        fold: set(races.loc[races["model_fold"].eq(fold), "rcDate"])
        for fold in ["train", "calibration", "test"]
    }
    for left, right in [("train", "calibration"), ("train", "test"), ("calibration", "test")]:
        if dates_by_fold[left] & dates_by_fold[right]:
            raise ValueError(f"Race dates overlap between {left} and {right}")

    audit = {
        "race_count": int(len(races)),
        "entry_count": int(len(entries)),
        "folds": fold_summary,
        "boundaries": {
            "train_end": TRAIN_END,
            "calibration_start": CALIBRATION_START,
            "calibration_end": CALIBRATION_END,
            "test_start": TEST_START,
        },
        "same_race_single_fold": True,
        "same_date_single_model_fold": True,
    }
    return races, audit


def write_outputs(
    races: pd.DataFrame,
    audit: dict[str, Any],
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    races.to_csv(output_path, index=False, encoding="utf-8-sig")

    manifest = {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(DEFAULT_SEOUL_INTERIM_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(DEFAULT_SEOUL_INTERIM_PATH),
        },
        "output": {
            "path": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": int(output_path.stat().st_size),
            "sha256": sha256_file(output_path),
            "unit": "one row per race_id",
        },
        "policy": {
            "split_method": "fixed chronological race-date boundaries",
            "excluded_races": "retain with model_fold=excluded; never remove individual entries",
            "test_usage": "single final evaluation only",
        },
        "observed": audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    races, audit = build_split_frame(load_seoul_interim())
    manifest = write_outputs(races, audit, args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
