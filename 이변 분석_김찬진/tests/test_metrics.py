from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.metrics import lift_at, top_mask  # noqa: E402


class MetricTest(unittest.TestCase):
    def test_top_mask_size(self) -> None:
        mask = top_mask(np.arange(100), 0.10)
        self.assertEqual(mask.sum(), 10)
        self.assertTrue(mask[-1])

    def test_perfect_top_lift(self) -> None:
        y = np.array([0] * 90 + [1] * 10)
        score = np.arange(100)
        self.assertAlmostEqual(lift_at(y, score, 0.10), 10.0)


if __name__ == "__main__":
    unittest.main()
