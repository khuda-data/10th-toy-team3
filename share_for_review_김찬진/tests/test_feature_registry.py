from __future__ import annotations

import unittest

from src.data.load_interim import load_seoul_interim
from src.features.registry import (
    assert_feature_list,
    load_feature_registry,
    select_premarket_features,
    validate_registry_columns,
)


class FeatureRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_seoul_interim()
        cls.registry = load_feature_registry()

    def test_all_columns_are_registered_exactly_once(self) -> None:
        validate_registry_columns(self.frame.columns)
        self.assertEqual(len(self.registry["features"]), 162)

    def test_critical_roles(self) -> None:
        expected = {
            "win": "TARGET",
            "longshot_win": "TARGET",
            "ord": "POST_RACE",
            "q_market": "MARKET",
            "winOdds": "MARKET",
            "race_id": "ID",
            "rcDate": "SPLIT",
            "legacy_upset_B": "LEGACY",
            "eligible_primary": "CONTROL",
            "train_runs_14": "PRE_RACE",
        }
        for column, role in expected.items():
            self.assertEqual(self.registry["features"][column]["role"], role)

    def test_premarket_selection_contains_only_approved_features(self) -> None:
        selected = select_premarket_features(self.frame.columns)
        self.assertGreater(len(selected), 0)
        self.assertTrue(all(self.registry["features"][c]["role"] == "PRE_RACE" for c in selected))
        assert_feature_list(selected)

    def test_leakage_columns_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_feature_list(["train_runs_14", "win", "q_market", "race_id"])

    def test_unknown_column_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_feature_list(["future_unknown_feature"])


if __name__ == "__main__":
    unittest.main()
