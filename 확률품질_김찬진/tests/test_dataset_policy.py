from __future__ import annotations

import unittest

from src.data.dataset_policy import assert_dataset_allowed, dataset_record


class DatasetPolicyTest(unittest.TestCase):
    def test_canonical_datasets_are_allowed(self) -> None:
        for path in [
            "data/raw/final.csv.gz",
            "data/interim/seoul_entries.csv.gz",
            "data/interim/split_manifest.csv",
        ]:
            self.assertEqual(dataset_record(path)["status"], "canonical")
            assert_dataset_allowed(path)

    def test_v5_to_v8_are_forbidden(self) -> None:
        for version in [
            "v5_base_no_outlier",
            "v6_standard_no_outlier",
            "v7_minmax_no_outlier",
            "v8_robust_no_outlier",
        ]:
            path = f"전처리 데이터셋/{version}/train.csv"
            self.assertEqual(dataset_record(path)["status"], "forbidden_model_input")
            with self.assertRaises(ValueError):
                assert_dataset_allowed(path)

    def test_v1_to_v4_are_legacy_only(self) -> None:
        for version in ["v1_base", "v2_standard", "v3_minmax", "v4_robust"]:
            path = f"전처리 데이터셋/{version}/train.csv"
            self.assertEqual(dataset_record(path)["status"], "legacy_read_only")
            with self.assertRaises(ValueError):
                assert_dataset_allowed(path)

    def test_unregistered_dataset_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            assert_dataset_allowed("unknown.csv")


if __name__ == "__main__":
    unittest.main()
