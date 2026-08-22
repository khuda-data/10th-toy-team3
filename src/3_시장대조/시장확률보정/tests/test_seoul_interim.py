from __future__ import annotations

import json
import unittest

import numpy as np

from src.data.build_seoul_interim import DEFAULT_MANIFEST_PATH
from src.data.load_interim import DEFAULT_SEOUL_INTERIM_PATH, load_seoul_interim
from src.data.load_raw import load_raw
from src.data.validate_schema import sha256_file


class SeoulInterimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_seoul_interim()
        cls.raw_seoul = load_raw().loc[lambda x: x["meet"].eq("서울")]
        cls.manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_expected_shape_and_scope(self) -> None:
        self.assertEqual(self.frame.shape, (32_888, 162))
        self.assertEqual(set(self.frame["meet"]), {"서울"})
        self.assertEqual(self.frame["race_id"].nunique(), 3_172)

    def test_no_seoul_rows_are_dropped_or_added(self) -> None:
        self.assertEqual(
            set(self.frame["entry_id"]),
            set(self.raw_seoul["entry_id"]),
        )

    def test_legacy_columns_are_isolated(self) -> None:
        self.assertTrue(
            {"legacy_fold", "legacy_upset_A", "legacy_upset_B", "legacy_upset"}
            .issubset(self.frame.columns)
        )
        self.assertTrue(
            {"fold", "upset_A", "upset_B", "upset"}.isdisjoint(self.frame.columns)
        )

    def test_corrected_longshot_win_contract(self) -> None:
        expected = (
            self.frame["pop_pct"].ge(0.50) & self.frame["win"].eq(1)
        ).astype("int8")
        self.assertTrue(self.frame["longshot_win"].eq(expected).all())
        self.assertEqual(int(self.frame["longshot_win"].sum()), 495)
        self.assertEqual(int(self.frame["legacy_upset_B"].sum()), 2_400)

    def test_market_probability_contract(self) -> None:
        sums = self.frame.groupby("race_id")["q_market"].sum().to_numpy()
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-12))
        self.assertLess((self.frame["q_market"] - self.frame["q"]).abs().max(), 1e-12)

    def test_race_status_policy(self) -> None:
        status = (
            self.frame[["race_id", "race_status"]]
            .drop_duplicates()["race_status"]
            .value_counts()
            .to_dict()
        )
        self.assertEqual(status, {"normal": 3_167, "dead_heat": 4, "no_winner": 1})
        self.assertTrue(
            self.frame["eligible_primary"].eq(self.frame["race_status"].eq("normal")).all()
        )
        self.assertEqual(
            self.frame.loc[self.frame["eligible_primary"], "race_id"].nunique(),
            3_167,
        )

    def test_manifest_matches_output(self) -> None:
        self.assertEqual(
            sha256_file(DEFAULT_SEOUL_INTERIM_PATH),
            self.manifest["output"]["sha256"],
        )
        self.assertEqual(self.manifest["observed"]["rows"], 32_888)
        self.assertEqual(self.manifest["observed"]["eligible_primary_races"], 3_167)


if __name__ == "__main__":
    unittest.main()
