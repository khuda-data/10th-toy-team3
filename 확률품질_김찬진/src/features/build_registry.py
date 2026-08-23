"""Generate the reviewed static feature registry for Seoul interim columns."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.load_interim import DEFAULT_SEOUL_INTERIM_PATH, load_seoul_interim
from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "manifests" / "feature_registry.json"

ROLE_COLUMNS = {
    "ID": {
        "hrName", "hrNo", "jkName", "jkNo", "meet", "owName", "owNo",
        "trName", "trNo", "meet_cd", "race_id", "entry_id",
    },
    "SPLIT": {"rcDate"},
    "TARGET": {"win", "longshot_win"},
    "POST_RACE": {"ord", "fin_rank", "fin_pct", "place", "resid"},
    "LEGACY": {"legacy_fold", "legacy_upset_A", "legacy_upset_B", "legacy_upset"},
    "CONTROL": {"race_winner_rows", "race_status", "eligible_primary"},
    "MARKET": {
        "plcOdds", "winOdds", "p_raw", "book_sum", "takeout", "q",
        "logit_q", "log_q", "pop_rank", "pop_pct", "is_fav",
        "winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse",
        "pl_harville", "pl_disc", "q_plc", "gap_h", "gap_d",
        "q_market", "longshot",
    },
}

ROLE_AVAILABLE_AT = {
    "ID": "identity",
    "SPLIT": "before_race",
    "PRE_RACE": "before_race",
    "MARKET": "market_close",
    "POST_RACE": "after_race",
    "TARGET": "after_race",
    "LEGACY": "not_approved",
    "CONTROL": "pipeline_control",
}


def build_registry() -> dict:
    frame = load_seoul_interim()
    columns = list(frame.columns)
    assigned: dict[str, str] = {}
    for role, names in ROLE_COLUMNS.items():
        for name in names:
            if name in assigned:
                raise ValueError(f"Column assigned to multiple roles: {name}")
            assigned[name] = role

    missing_reviewed = sorted(set(assigned) - set(columns))
    if missing_reviewed:
        raise ValueError(f"Reviewed registry columns absent from dataset: {missing_reviewed}")

    # All remaining columns were reviewed as pre-race inputs at registry creation
    # time. The emitted JSON is static; future unknown columns fail validation
    # until this builder is deliberately rerun and reviewed.
    for name in columns:
        assigned.setdefault(name, "PRE_RACE")

    entries = {}
    for name in columns:
        role = assigned[name]
        entries[name] = {
            "role": role,
            "available_at": ROLE_AVAILABLE_AT[role],
            "allowed_premarket_feature": role == "PRE_RACE",
            "allowed_market_offset_feature": role == "PRE_RACE",
            "usage": "offset_only" if name == "q_market" else "standard",
        }

    schema_hash = hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()
    counts = {}
    for entry in entries.values():
        counts[entry["role"]] = counts.get(entry["role"], 0) + 1

    return {
        "registry_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(DEFAULT_SEOUL_INTERIM_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(DEFAULT_SEOUL_INTERIM_PATH),
            "column_count": len(columns),
            "ordered_columns_sha256": schema_hash,
        },
        "policy": {
            "default_for_unknown_column": "reject",
            "premarket_model_roles": ["PRE_RACE"],
            "market_offset": "q_market is supplied only as offset, not a standard feature",
        },
        "role_counts": counts,
        "features": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    registry = build_registry()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **registry["role_counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
