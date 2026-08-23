"""Integrity checks for revised_v5 before training."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


DATA = Path("data/revised_v5")
FILES = {"train": "train_revised_v5.csv", "valid": "valid_revised_v5.csv", "test": "test_revised_v5.csv"}


def main() -> None:
    frames = {name: pd.read_csv(DATA / file, low_memory=False) for name, file in FILES.items()}
    schema = len({tuple(frame.columns) for frame in frames.values()}) == 1
    details, failures, ranges = {}, [], []
    for name, frame in frames.items():
        winner_errors = int((frame.groupby("race_id")["win"].sum() != 1).sum())
        duplicates = int(frame["entry_id"].duplicated().sum())
        ranges.append((int(frame.rcDate.min()), int(frame.rcDate.max())))
        details[name] = {"rows": len(frame), "races": int(frame.race_id.nunique()), "columns": len(frame.columns), "null_cells": int(frame.isna().sum().sum()), "duplicate_entry_ids": duplicates, "invalid_winner_races": winner_errors, "date_range": ranges[-1]}
        if duplicates or winner_errors:
            failures.append(name)
    chronology = ranges[0][1] < ranges[1][0] < ranges[2][0]
    raw = {"sex", "weather", "rcDay", "budam", "born", "jkName", "trName", "owName", "rank", "tool_set", "track", "meet", "fold", "wgBudamBigo"}
    remaining = sorted(raw & set(frames["train"].columns))
    expected = {"te_jkName", "te_trName", "te_owName", "te_rank"}
    missing = sorted(expected - set(frames["train"].columns))
    report = {"dataset": "revised_v5", "schema_consistent": schema, "chronology_strict": chronology, "splits": details, "raw_string_columns_remaining": remaining, "missing_target_encoded_columns": missing, "training_ready": schema and chronology and not failures and not remaining and not missing, "blocking_issues": failures}
    (DATA / "integrity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["training_ready"]:
        raise SystemExit("v5 integrity validation failed")


if __name__ == "__main__":
    main()
