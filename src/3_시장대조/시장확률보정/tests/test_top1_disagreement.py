from __future__ import annotations

import json
import unittest

import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file
from src.evaluation.top1_disagreement import (
    build_race_disagreement_table,
    build_segment_summary,
)


class Top1DisagreementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r1", "r2", "r2"],
                "entry_id": ["r1_01", "r1_02", "r1_03", "r2_01", "r2_02"],
                "rcDate": [20250101] * 3 + [20250102] * 2,
                "win": [1, 0, 0, 0, 1],
                "q_market": [0.5, 0.3, 0.2, 0.6, 0.4],
                "p_model_race": [0.2, 0.5, 0.3, 0.5, 0.5],
                "rcDist": [1200] * 5,
                "rank": ["국6"] * 5,
                "track": ["건조"] * 5,
                "weather": ["맑음"] * 5,
                "feature_a": [1.0, None, 3.0, 1.0, 2.0],
            }
        )

    def test_correctness_cases_and_deterministic_tie_break(self) -> None:
        races = build_race_disagreement_table(
            self.frame, source="synthetic", feature_columns=["feature_a"]
        ).set_index("race_id")
        self.assertEqual(races.loc["r1", "correctness_case"], "market_only_correct")
        self.assertTrue(races.loc["r1", "top1_disagreement"])
        self.assertEqual(races.loc["r2", "correctness_case"], "both_wrong")
        self.assertEqual(races.loc["r2", "model_top_entry_id"], "r2_01")
        self.assertFalse(races.loc["r2", "top1_disagreement"])

    def test_segment_summary_preserves_correctness_arithmetic(self) -> None:
        races = build_race_disagreement_table(self.frame, source="synthetic")
        summary = build_segment_summary(races)
        overall = summary.loc[
            summary["source"].eq("combined") & summary["segment_name"].eq("all")
        ].iloc[0]
        self.assertEqual(overall["races"], 2)
        self.assertEqual(overall["market_correct"], 1)
        self.assertEqual(overall["model_correct"], 0)
        self.assertEqual(overall["model_minus_market_correct"], -1)

    def test_invalid_probability_simplex_is_rejected(self) -> None:
        invalid = self.frame.copy()
        invalid.loc[invalid["race_id"].eq("r1"), "p_model_race"] = 0.1
        with self.assertRaises(ValueError):
            build_race_disagreement_table(invalid, source="synthetic")

    def test_duplicate_entry_is_rejected(self) -> None:
        invalid = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_race_disagreement_table(invalid, source="synthetic")

    def test_stage_22_artifacts_use_only_oof_and_calibration(self) -> None:
        report_path = (
            PROJECT_ROOT
            / "reports"
            / "experiments"
            / "stage_22_top1_disagreement.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            report["data_policy"]["opened_final_test"],
            "not_loaded_not_aggregated_forbidden_for_selection",
        )
        self.assertLessEqual(report["data_policy"]["max_rcDate"], 20251227)
        self.assertEqual(report["overall"]["train_oof"]["races"], 770)
        self.assertEqual(report["overall"]["calibration"]["races"], 641)
        self.assertEqual(report["overall"]["combined"]["races"], 1411)

        races = pd.read_csv(
            PROJECT_ROOT / "data" / "analysis" / "stage_22_race_disagreements.csv.gz"
        )
        self.assertEqual(set(races["source"]), {"train_oof", "calibration"})
        self.assertEqual(races["race_id"].nunique(), 1411)
        self.assertLessEqual(int(races["rcDate"].max()), 20251227)
        self.assertEqual(
            races["correctness_case"].value_counts().to_dict(),
            {
                "both_wrong": 764,
                "both_correct": 289,
                "market_only_correct": 242,
                "model_only_correct": 116,
            },
        )

    def test_stage_22_output_hashes_match_report(self) -> None:
        report = json.loads(
            (
                PROJECT_ROOT
                / "reports"
                / "experiments"
                / "stage_22_top1_disagreement.json"
            ).read_text(encoding="utf-8")
        )
        for key, relative in report["outputs"].items():
            self.assertEqual(
                sha256_file(PROJECT_ROOT / relative),
                report["output_sha256"][key],
            )


if __name__ == "__main__":
    unittest.main()
