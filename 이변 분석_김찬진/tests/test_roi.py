from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.roi import percentile_roi_tables  # noqa: E402


class RoiTest(unittest.TestCase):
    def test_realized_roi_formula(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": ["r1", "r2"],
                "target": [1, 0],
                "plcOdds": [3.0, 2.0],
                "score": [0.9, 0.1],
                "calibrated_probability": [0.5, 0.2],
            }
        )
        cumulative, _ = percentile_roi_tables(frame)
        total = cumulative.loc[cumulative["range"].eq("top_100pct")].iloc[0]
        self.assertAlmostEqual(total["realized_roi"], 0.5)


if __name__ == "__main__":
    unittest.main()
