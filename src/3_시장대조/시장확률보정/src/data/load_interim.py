"""Load canonical interim datasets without losing identifier formatting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_raw import ID_DTYPES, ID_WIDTHS, PROJECT_ROOT


DEFAULT_SEOUL_INTERIM_PATH = (
    PROJECT_ROOT / "data" / "interim" / "seoul_entries.csv.gz"
)


def load_seoul_interim(path: str | Path | None = None) -> pd.DataFrame:
    interim_path = Path(path) if path is not None else DEFAULT_SEOUL_INTERIM_PATH
    if not interim_path.is_file():
        raise FileNotFoundError(f"Seoul interim dataset not found: {interim_path}")

    frame = pd.read_csv(
        interim_path,
        compression="infer",
        dtype=ID_DTYPES,
        low_memory=False,
    )
    for column, width in ID_WIDTHS.items():
        if column in frame.columns:
            frame[column] = frame[column].str.zfill(width)
    return frame
