"""Validate the canonical raw dataset and its frozen manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from src.data.load_raw import DEFAULT_RAW_PATH, PROJECT_ROOT, load_raw


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "raw_manifest.json"

EXPECTED_ROWS = 56_648
EXPECTED_COLUMNS = 156
EXPECTED_DATE_MIN = 20_230_805
EXPECTED_DATE_MAX = 20_260_809
EXPECTED_RACES = 5_361
EXPECTED_MARKETS = {"서울": 32_888, "부경": 23_760}

REQUIRED_COLUMNS = {
    "entry_id",
    "race_id",
    "rcDate",
    "meet",
    "win",
    "winOdds",
    "hrNo",
    "jkNo",
    "trNo",
    "owNo",
}


def _hash_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return _hash_stream(stream)


def sha256_decompressed_gzip(path: str | Path) -> str:
    with gzip.open(path, "rb") as stream:
        return _hash_stream(stream)


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    winners_per_race = frame.groupby("race_id", dropna=False)["win"].sum()
    market_counts = {
        str(name): int(count)
        for name, count in frame["meet"].value_counts(dropna=False).items()
    }
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "date_min": int(frame["rcDate"].min()),
        "date_max": int(frame["rcDate"].max()),
        "race_count": int(frame["race_id"].nunique(dropna=True)),
        "entry_id_unique": bool(frame["entry_id"].is_unique),
        "entry_id_missing": int(frame["entry_id"].isna().sum()),
        "markets": market_counts,
        "win_values": sorted(int(value) for value in frame["win"].dropna().unique()),
        "races_one_winner": int((winners_per_race == 1).sum()),
        "races_multiple_winners": int((winners_per_race > 1).sum()),
        "races_no_winner": int((winners_per_race == 0).sum()),
    }


def validate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return {"errors": errors, "warnings": warnings, "summary": {}}

    summary = summarize_frame(frame)
    expected = {
        "rows": EXPECTED_ROWS,
        "columns": EXPECTED_COLUMNS,
        "date_min": EXPECTED_DATE_MIN,
        "date_max": EXPECTED_DATE_MAX,
        "race_count": EXPECTED_RACES,
        "markets": EXPECTED_MARKETS,
    }
    for field, expected_value in expected.items():
        if summary[field] != expected_value:
            errors.append(
                f"Unexpected {field}: {summary[field]!r} != {expected_value!r}"
            )

    if not summary["entry_id_unique"]:
        errors.append("entry_id is not unique")
    if summary["entry_id_missing"]:
        errors.append(f"entry_id has {summary['entry_id_missing']} missing values")
    if summary["win_values"] != [0, 1]:
        errors.append(f"win must be binary, found {summary['win_values']}")
    if frame["race_id"].isna().any():
        errors.append("race_id contains missing values")
    if frame["rcDate"].isna().any():
        errors.append("rcDate contains missing values")
    if (frame["winOdds"] <= 0).any():
        errors.append("winOdds contains a non-positive value")

    if summary["races_multiple_winners"]:
        warnings.append(
            f"{summary['races_multiple_winners']} races have multiple winner rows; "
            "retain them in raw data and define dead-heat policy downstream."
        )
    if summary["races_no_winner"]:
        warnings.append(
            f"{summary['races_no_winner']} races have no winner row; retain them "
            "in raw data and reject or resolve the whole race downstream."
        )

    return {"errors": errors, "warnings": warnings, "summary": summary}


def validate_raw_file(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    raw_path = Path(raw_path)
    manifest_path = Path(manifest_path)
    frame_result = validate_frame(load_raw(raw_path))
    errors = list(frame_result["errors"])
    warnings = list(frame_result["warnings"])

    compressed_hash = sha256_file(raw_path)
    decompressed_hash = sha256_decompressed_gzip(raw_path)

    if not manifest_path.is_file():
        errors.append(f"Manifest not found: {manifest_path}")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_compressed = manifest.get("file", {}).get("sha256")
        expected_decompressed = manifest.get("file", {}).get("decompressed_sha256")
        if compressed_hash != expected_compressed:
            errors.append("Compressed SHA-256 does not match raw manifest")
        if decompressed_hash != expected_decompressed:
            errors.append("Decompressed SHA-256 does not match raw manifest")
        if frame_result["summary"] != manifest.get("observed"):
            errors.append("Observed dataset summary does not match raw manifest")

    return {
        "ok": not errors,
        "raw_path": str(raw_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "compressed_sha256": compressed_hash,
        "decompressed_sha256": decompressed_hash,
        "errors": errors,
        "warnings": warnings,
        "summary": frame_result["summary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    result = validate_raw_file(args.raw, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
