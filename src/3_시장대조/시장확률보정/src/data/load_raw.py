"""Canonical loader for the immutable raw horse-racing dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "final.csv.gz"

# Registration numbers and compound keys must remain strings so leading zeroes
# are never discarded by pandas' type inference.
ID_DTYPES = {
    "entry_id": "string",
    "race_id": "string",
    "hrNo": "string",
    "jkNo": "string",
    "trNo": "string",
    "owNo": "string",
}

# The source CSV stores most Seoul horse numbers as five characters even though
# the documented registration-number format is seven characters. Canonical
# loading restores the omitted leading zeroes without altering the raw file.
ID_WIDTHS = {
    "hrNo": 7,
    "jkNo": 6,
    "trNo": 6,
    "owNo": 6,
}


def load_raw(
    path: str | Path | None = None,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the canonical immutable raw dataset.

    Parameters
    ----------
    path:
        Optional alternative CSV/CSV.GZ path. Defaults to data/raw/final.csv.gz.
    columns:
        Optional subset of columns to load.
    """

    raw_path = Path(path) if path is not None else DEFAULT_RAW_PATH
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    usecols = list(columns) if columns is not None else None
    dtype = {
        name: value
        for name, value in ID_DTYPES.items()
        if usecols is None or name in usecols
    }

    frame = pd.read_csv(
        raw_path,
        compression="infer",
        usecols=usecols,
        dtype=dtype,
        low_memory=False,
    )

    for column, width in ID_WIDTHS.items():
        if column in frame.columns:
            frame[column] = frame[column].str.zfill(width)

    return frame
