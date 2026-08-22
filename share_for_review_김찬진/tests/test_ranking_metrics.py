from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file
from src.evaluation.ranking_metrics import rank_entries, ranking_metrics


class RankingMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r1", "r2", "r2"],
                "entry_id": ["r1_01", "r1_02", "r1_03", "r2_01", "r2_02"],
                "win": [0, 1, 0, 1, 0],
                "q_market": [0.2, 0.5, 0.3, 0.4, 0.6],
            }
        )

    def test_unique_tie_break_uses_market_then_entry_id(self) -> None:
        scores = np.asarray([0.7, 0.7, 0.1, 0.5, 0.5])
        ranked = rank_entries(self.frame, scores).set_index("entry_id")
        self.assertEqual(ranked.loc["r1_02", "rank_position"], 1)
        self.assertEqual(ranked.loc["r2_02", "rank_position"], 1)
        self.assertEqual(ranked.groupby("race_id")["rank_position"].min().tolist(), [1, 1])

    def test_ranking_metrics_use_one_winner_rank_per_race(self) -> None:
        metrics = ranking_metrics(self.frame, [0.1, 0.9, 0.2, 0.8, 0.2])
        self.assertEqual(metrics["top1_correct"], 2)
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["hit_at_3"], 1.0)
        self.assertEqual(metrics["mean_reciprocal_rank"], 1.0)
        self.assertEqual(metrics["winner_mean_rank"], 1.0)

    def test_nonfinite_scores_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ranking_metrics(self.frame, [0.1, np.nan, 0.2, 0.8, 0.2])

    def test_stage_24_outputs_are_rank_scores_not_probabilities(self) -> None:
        report_path = PROJECT_ROOT / "reports" / "experiments" / "stage_24_ranker.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["data_policy"]["opened_final_test"], "not_loaded_not_evaluated")
        self.assertLessEqual(report["data_policy"]["max_rcDate"], 20251227)
        self.assertEqual(report["model"]["parameters"]["objective"], "rank:pairwise")
        self.assertEqual(report["model"]["feature_count"], 112)
        self.assertEqual(report["model"]["probability_status"], "not_converted_until_stage_25")
        self.assertEqual(
            report["evaluation_policy"]["logloss_brier_status"],
            "not_applicable_to_raw_ranking_scores",
        )
        self.assertEqual(len(report["walk_forward"]), 4)
        self.assertEqual(report["evaluation"]["train_oof_pooled"]["R2_pairwise_ranker"]["races"], 770)
        self.assertEqual(report["evaluation"]["calibration"]["R2_pairwise_ranker"]["races"], 641)
        self.assertEqual(
            report["evaluation"]["train_oof_pooled"]["R2_pairwise_ranker"]["top1_correct"],
            216,
        )
        self.assertEqual(
            report["evaluation"]["calibration"]["R2_pairwise_ranker"]["top1_correct"],
            193,
        )
        self.assertEqual(
            report["evaluation"]["calibration"]["comparison"]["r2_minus_m2"]["top1_correct"],
            3,
        )
        self.assertEqual(
            report["evaluation"]["calibration"]["comparison"]["r2_minus_market"]["top1_correct"],
            -54,
        )
        for key, relative in report["outputs"].items():
            self.assertEqual(sha256_file(PROJECT_ROOT / relative), report["output_sha256"][key])


if __name__ == "__main__":
    unittest.main()
