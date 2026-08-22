"""Build the canonical Seoul-only interim dataset from immutable raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from src.data.load_interim import DEFAULT_SEOUL_INTERIM_PATH
from src.data.load_raw import DEFAULT_RAW_PATH, PROJECT_ROOT, load_raw
from src.data.validate_schema import sha256_file


DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "manifests" / "seoul_interim_manifest.json"
)

EXPECTED_ROWS = 32_888
EXPECTED_RACES = 3_172
EXPECTED_COLUMNS = 162

LEGACY_RENAMES = {
    "fold": "legacy_fold",
    "upset_A": "legacy_upset_A",
    "upset_B": "legacy_upset_B",
    "upset": "legacy_upset",
}


def _hash_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def build_seoul_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(set(LEGACY_RENAMES) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing legacy columns required for canonical rename: {missing}")

    frame = raw.loc[raw["meet"].eq("서울")].copy()
    frame = frame.rename(columns=LEGACY_RENAMES)

    p_raw_market = 1.0 / frame["winOdds"]
    book_sum_market = p_raw_market.groupby(frame["race_id"]).transform("sum")
    frame["q_market"] = p_raw_market / book_sum_market

    # q is retained only for source-audit compatibility. q_market is the
    # canonical market probability used by all new models.
    q_max_abs_diff = float((frame["q_market"] - frame["q"]).abs().max())

    frame["longshot"] = (frame["pop_pct"] >= 0.50).astype("int8")
    frame["longshot_win"] = (
        frame["longshot"].eq(1) & frame["win"].eq(1)
    ).astype("int8")

    frame["race_winner_rows"] = (
        frame.groupby("race_id")["win"].transform("sum").astype("int8")
    )
    frame["race_status"] = np.select(
        [frame["race_winner_rows"].eq(0), frame["race_winner_rows"].gt(1)],
        ["no_winner", "dead_heat"],
        default="normal",
    )
    frame["eligible_primary"] = frame["race_status"].eq("normal")

    frame = frame.sort_values(
        ["rcDate", "race_id", "chulNo", "entry_id"],
        kind="stable",
    ).reset_index(drop=True)

    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"Unexpected Seoul shape: {frame.shape} != "
            f"({EXPECTED_ROWS}, {EXPECTED_COLUMNS})"
        )
    if frame["race_id"].nunique() != EXPECTED_RACES:
        raise ValueError("Unexpected number of Seoul races")
    if not frame["entry_id"].is_unique:
        raise ValueError("entry_id is not unique in Seoul interim data")
    if not np.allclose(
        frame.groupby("race_id")["q_market"].sum().to_numpy(),
        1.0,
        atol=1e-12,
    ):
        raise ValueError("q_market does not sum to 1 within every race")
    if not frame["longshot_win"].eq(
        (frame["pop_pct"].ge(0.50) & frame["win"].eq(1)).astype("int8")
    ).all():
        raise ValueError("longshot_win label contract is violated")

    race_status = (
        frame[["race_id", "race_status"]]
        .drop_duplicates()
        ["race_status"]
        .value_counts()
        .to_dict()
    )
    row_status = frame["race_status"].value_counts().to_dict()
    bad_races = []
    for race_id, race in frame.loc[frame["race_status"].ne("normal")].groupby("race_id"):
        bad_races.append(
            {
                "race_id": str(race_id),
                "rcDate": int(race["rcDate"].iloc[0]),
                "rows": int(len(race)),
                "winner_rows": int(race["win"].sum()),
                "status": str(race["race_status"].iloc[0]),
            }
        )

    audit = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "race_count": int(frame["race_id"].nunique()),
        "entry_id_unique": bool(frame["entry_id"].is_unique),
        "date_min": int(frame["rcDate"].min()),
        "date_max": int(frame["rcDate"].max()),
        "q_market_max_abs_diff_from_legacy_q": q_max_abs_diff,
        "race_status_counts": {str(k): int(v) for k, v in race_status.items()},
        "row_status_counts": {str(k): int(v) for k, v in row_status.items()},
        "eligible_primary_rows": int(frame["eligible_primary"].sum()),
        "eligible_primary_races": int(
            frame.loc[frame["eligible_primary"], "race_id"].nunique()
        ),
        "longshot_rows": int(frame["longshot"].sum()),
        "longshot_win_rows": int(frame["longshot_win"].sum()),
        "legacy_upset_B_positive_rows": int(frame["legacy_upset_B"].sum()),
        "non_normal_races": bad_races,
    }
    return frame, audit


def write_outputs(
    frame: pd.DataFrame,
    audit: dict[str, Any],
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )

    manifest = {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(DEFAULT_RAW_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(DEFAULT_RAW_PATH),
        },
        "output": {
            "path": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "format": "gzip-compressed UTF-8-SIG CSV",
            "bytes": int(output_path.stat().st_size),
            "sha256": sha256_file(output_path),
        },
        "transformations": [
            "Filter meet == 서울 without dropping any Seoul entry rows.",
            "Rename fold and upset labels to legacy_* names.",
            "Recompute canonical q_market from winOdds; retain legacy p_raw, book_sum and q for source audit.",
            "Create corrected longshot and longshot_win labels.",
            "Mark normal, dead_heat and no_winner races without row deletion.",
        ],
        "primary_policy": {
            "target": "win",
            "market_probability": "q_market",
            "eligible_condition": "eligible_primary == true",
            "non_normal_race_policy": "retain in interim; exclude whole race from primary training/evaluation",
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
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SEOUL_INTERIM_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    frame, audit = build_seoul_frame(load_raw(args.raw))
    manifest = write_outputs(frame, audit, args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
