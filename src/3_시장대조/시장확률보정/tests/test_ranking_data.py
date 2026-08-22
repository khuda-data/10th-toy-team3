from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.ranking_data import (
    build_ranking_dataset,
    build_ranking_manifests,
    validate_ranking_manifests,
)
from src.data.validate_schema import sha256_file


class RankingDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "race_id": ["r2", "r1", "r1", "r2", "r1"],
                "entry_id": ["r2_02", "r1_03", "r1_01", "r2_01", "r1_02"],
                "rcDate": [20250102, 20250101, 20250101, 20250102, 20250101],
                "win": [1, 0, 1, 0, 0],
                "age": [4, 3, 3, 4, 3],
                "rating": [40.0, None, 35.0, 42.0, 38.0],
            }
        )

    def test_builder_sorts_groups_and_aligns_relevance(self) -> None:
        dataset = build_ranking_dataset(
            self.frame, feature_names=("age", "rating")
        )
        self.assertEqual(dataset.group_ids, ("r1", "r2"))
        np.testing.assert_array_equal(dataset.group_sizes, [3, 2])
        np.testing.assert_array_equal(dataset.relevance, [1, 0, 0, 0, 1])
        self.assertEqual(dataset.features.columns.tolist(), ["age", "rating"])

    def test_persisted_group_intervals_are_exact(self) -> None:
        dataset = build_ranking_dataset(
            self.frame, feature_names=("age", "rating")
        )
        entries, groups = build_ranking_manifests(dataset, model_fold="train")
        validate_ranking_manifests(entries, groups)
        self.assertEqual(groups["row_start"].tolist(), [0, 3])
        self.assertEqual(groups["row_stop_exclusive"].tolist(), [3, 5])
        self.assertEqual(groups["winner_position_in_group"].tolist(), [0, 1])

    def test_forbidden_market_feature_is_rejected(self) -> None:
        frame = self.frame.assign(q_market=0.5)
        with self.assertRaises(ValueError):
            build_ranking_dataset(frame, feature_names=("age", "q_market"))

    def test_invalid_winner_count_is_rejected(self) -> None:
        invalid = self.frame.copy()
        invalid.loc[invalid["race_id"].eq("r1"), "win"] = 0
        with self.assertRaises(ValueError):
            build_ranking_dataset(invalid, feature_names=("age", "rating"))

    def test_stage_23_manifest_and_outputs_are_locked(self) -> None:
        manifest_path = (
            PROJECT_ROOT / "data" / "manifests" / "ranking_dataset_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["data_policy"]["included_folds"], ["train", "calibration"])
        self.assertEqual(manifest["data_policy"]["opened_final_test"], "not_included")
        self.assertEqual(manifest["feature_contract"]["feature_count"], 112)
        self.assertEqual(manifest["feature_contract"]["allowed_roles"], ["PRE_RACE"])
        self.assertFalse(manifest["feature_contract"]["market_features_in_ranker"])
        self.assertEqual(manifest["observed"]["total_rows"], 26199)
        self.assertEqual(manifest["observed"]["total_groups"], 2532)
        self.assertTrue(manifest["observed"]["group_sizes_sum_to_rows"])
        self.assertTrue(manifest["observed"]["all_group_relevance_sums_one"])
        self.assertTrue(all(row["strictly_chronological"] for row in manifest["walk_forward"]))
        self.assertFalse(any(row["group_overlap"] for row in manifest["walk_forward"]))
        for output in manifest["outputs"].values():
            path = PROJECT_ROOT / output["path"]
            self.assertEqual(path.stat().st_size, output["bytes"])
            self.assertEqual(sha256_file(path), output["sha256"])


if __name__ == "__main__":
    unittest.main()
