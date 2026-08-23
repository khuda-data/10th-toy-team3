from __future__ import annotations

import json
import unittest

import pandas as pd

from src.data.build_splits import DEFAULT_MANIFEST_PATH, DEFAULT_OUTPUT_PATH
from src.data.load_interim import load_seoul_interim
from src.data.validate_schema import sha256_file


class SplitManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.splits = pd.read_csv(DEFAULT_OUTPUT_PATH, dtype={"race_id": "string"})
        cls.entries = load_seoul_interim()
        cls.manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_one_row_per_race(self) -> None:
        self.assertEqual(len(self.splits), 3_172)
        self.assertTrue(self.splits["race_id"].is_unique)
        self.assertEqual(set(self.splits["race_id"]), set(self.entries["race_id"]))

    def test_fold_counts(self) -> None:
        self.assertEqual(
            self.splits["model_fold"].value_counts().to_dict(),
            {"train": 1_891, "calibration": 641, "test": 635, "excluded": 5},
        )

    def test_entry_counts_by_fold(self) -> None:
        joined = self.entries[["race_id", "entry_id"]].merge(
            self.splits[["race_id", "model_fold"]],
            on="race_id",
            validate="many_to_one",
        )
        self.assertEqual(
            joined["model_fold"].value_counts().to_dict(),
            {"train": 19_617, "test": 6_639, "calibration": 6_582, "excluded": 50},
        )

    def test_boundaries_and_date_separation(self) -> None:
        train = self.splits[self.splits["model_fold"].eq("train")]
        calibration = self.splits[self.splits["model_fold"].eq("calibration")]
        test = self.splits[self.splits["model_fold"].eq("test")]
        self.assertLessEqual(train["rcDate"].max(), 20_250_511)
        self.assertGreaterEqual(calibration["rcDate"].min(), 20_250_517)
        self.assertLessEqual(calibration["rcDate"].max(), 20_251_227)
        self.assertGreaterEqual(test["rcDate"].min(), 20_251_228)
        self.assertTrue(set(train["rcDate"]).isdisjoint(calibration["rcDate"]))
        self.assertTrue(set(train["rcDate"]).isdisjoint(test["rcDate"]))
        self.assertTrue(set(calibration["rcDate"]).isdisjoint(test["rcDate"]))

    def test_non_normal_races_are_excluded_whole(self) -> None:
        excluded = self.splits[self.splits["model_fold"].eq("excluded")]
        self.assertEqual(set(excluded["race_status"]), {"dead_heat", "no_winner"})
        self.assertFalse(excluded["eligible_primary"].any())

    def test_manifest_hash(self) -> None:
        self.assertEqual(sha256_file(DEFAULT_OUTPUT_PATH), self.manifest["output"]["sha256"])


if __name__ == "__main__":
    unittest.main()
