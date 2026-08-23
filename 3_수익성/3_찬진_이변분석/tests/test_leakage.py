from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.features import assert_no_leakage, select_features  # noqa: E402


class LeakageTest(unittest.TestCase):
    def test_leakage_columns_are_removed(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [3],
                "pop_pct": [0.8],
                "place": [1],
                "hrNo": [123],
                "hr_last_poppct": [0.5],
            }
        )
        core = select_features(frame, "core")
        assert_no_leakage(core, "core")
        self.assertEqual(core.columns.tolist(), ["age"])


if __name__ == "__main__":
    unittest.main()
