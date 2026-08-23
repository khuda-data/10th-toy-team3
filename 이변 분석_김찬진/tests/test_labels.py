from __future__ import annotations

import sys
import unittest
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.data import load_fold, validate_stored_label  # noqa: E402


class LabelDefinitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load_fold("train", include_outcomes=True)

    def test_darkhorse_matches_stored_label(self) -> None:
        self.assertEqual(validate_stored_label(self.train, "darkhorse"), 1.0)

    def test_favorite_bust_matches_stored_label(self) -> None:
        self.assertEqual(validate_stored_label(self.train, "favorite_bust"), 1.0)


if __name__ == "__main__":
    unittest.main()
