from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file
from src.evaluation.ranker_probability import ranking_scores_to_probabilities


class RankerProbabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r1", "r2", "r2"],
                "entry_id": ["a", "b", "c", "d", "e"],
            }
        )
        self.scores = np.asarray([2.0, 1.0, 0.0, -1.0, 1.0])

    def test_softmax_forms_independent_race_simplexes(self) -> None:
        probabilities = ranking_scores_to_probabilities(
            self.frame, self.scores, temperature=1.0
        )
        sums = pd.Series(probabilities).groupby(self.frame["race_id"]).sum()
        np.testing.assert_allclose(sums, 1.0)
        self.assertTrue((probabilities > 0).all())

    def test_translation_invariance_and_temperature_preserve_ranking(self) -> None:
        base = ranking_scores_to_probabilities(self.frame, self.scores, temperature=1.0)
        shifted = ranking_scores_to_probabilities(
            self.frame, self.scores + 100.0, temperature=1.0
        )
        sharp = ranking_scores_to_probabilities(
            self.frame, self.scores, temperature=0.5
        )
        flat = ranking_scores_to_probabilities(
            self.frame, self.scores, temperature=2.0
        )
        np.testing.assert_allclose(base, shifted)
        self.assertEqual(np.argmax(base[:3]), np.argmax(sharp[:3]))
        self.assertEqual(np.argmax(base[:3]), np.argmax(flat[:3]))
        self.assertGreater(sharp[:3].max(), base[:3].max())
        self.assertLess(flat[:3].max(), base[:3].max())

    def test_invalid_temperature_and_score_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ranking_scores_to_probabilities(self.frame, self.scores, temperature=0)
        invalid = self.scores.copy()
        invalid[0] = np.nan
        with self.assertRaises(ValueError):
            ranking_scores_to_probabilities(self.frame, invalid, temperature=1.0)

    def test_stage_25_policy_and_outputs(self) -> None:
        policy = json.loads(
            (
                PROJECT_ROOT
                / "data"
                / "manifests"
                / "ranker_temperature_policy.json"
            ).read_text(encoding="utf-8")
        )
        report = json.loads(
            (
                PROJECT_ROOT
                / "reports"
                / "experiments"
                / "stage_25_ranker_probability.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(policy["selection_fold"], "calibration")
        self.assertFalse(policy["opened_final_test_evaluated"])
        self.assertFalse(policy["top1_ranking_changes"])
        self.assertLessEqual(report["data_policy"]["max_rcDate"], 20251227)
        self.assertEqual(report["data_policy"]["opened_final_test"], "not_loaded_not_evaluated")
        self.assertEqual(len(report["temperature_grid"]), 63)
        self.assertEqual(policy["selected_temperature"], 0.65)
        self.assertEqual(
            report["evaluation"]["calibration"]["R2_ranker_probability"]["top1_correct"],
            193,
        )
        self.assertEqual(
            report["evaluation"]["calibration"]["R0_market"]["top1_correct"],
            247,
        )
        self.assertAlmostEqual(
            report["evaluation"]["calibration"]["R2_ranker_probability"]["race_log_loss"],
            1.9673195916135227,
        )
        self.assertLess(
            report["evaluation"]["calibration"]["r2_delta_vs_market"]["delta_logloss"],
            0.0,
        )
        self.assertGreater(
            report["evaluation"]["calibration"]["r2_delta_vs_m2"]["delta_logloss"],
            0.0,
        )
        for item in policy["outputs"]:
            self.assertEqual(sha256_file(PROJECT_ROOT / item["path"]), item["sha256"])
            frame = pd.read_csv(PROJECT_ROOT / item["path"])
            sums = frame.groupby("race_id")["p_ranker_race"].sum().to_numpy()
            self.assertTrue(np.allclose(sums, 1.0, atol=1e-9))
            self.assertTrue(frame["probability_status"].eq("race_softmax_calibrated").all())


if __name__ == "__main__":
    unittest.main()
