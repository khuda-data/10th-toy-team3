from __future__ import annotations

import json
import unittest

from src.data.load_raw import PROJECT_ROOT


class Top1ResearchPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = PROJECT_ROOT / "data" / "manifests" / "top1_research_policy.json"
        cls.policy = json.loads(path.read_text(encoding="utf-8"))

    def test_opened_final_test_is_descriptive_only(self) -> None:
        opened = self.policy["data_boundaries"]["opened_final_test"]
        self.assertEqual(opened["status"], "opened_descriptive_only")
        for forbidden in (
            "feature_selection",
            "gate_training",
            "threshold_selection",
            "candidate_selection",
            "official_success_claim",
        ):
            self.assertIn(forbidden, opened["forbidden_uses"])

    def test_future_holdout_is_predeclared_and_outcome_blind(self) -> None:
        holdout = self.policy["data_boundaries"]["future_holdout"]
        self.assertEqual(holdout["status"], "pending_data")
        self.assertEqual(holdout["anchor_date_exclusive"], 20260809)
        self.assertEqual(holdout["target_races"], 500)
        self.assertTrue(holdout["outcomes_sealed_until_candidate_freeze"])
        self.assertTrue(holdout["no_interim_model_changes"])

    def test_candidate_selection_is_constrained_before_top1(self) -> None:
        order = self.policy["candidate_selection_order"]
        self.assertLess(
            order.index("require_calibration_delta_logloss_gte_zero"),
            order.index("maximize_calibration_delta_top1"),
        )
        self.assertLess(
            order.index("require_calibration_delta_brier_gte_zero"),
            order.index("maximize_calibration_delta_top1"),
        )

    def test_official_top1_success_requires_paired_inference(self) -> None:
        success = self.policy["research_success_on_future_holdout"]
        self.assertEqual(success["delta_top1_gt"], 0.0)
        self.assertEqual(success["paired_bootstrap_delta_top1_ci_lower_gt"], 0.0)
        self.assertEqual(success["mcnemar_exact_two_sided_p_lt"], 0.05)
        self.assertGreaterEqual(success["delta_logloss_gte"], 0.0)
        self.assertGreaterEqual(success["delta_brier_gte"], 0.0)

    def test_probability_and_tie_break_contract_is_frozen(self) -> None:
        self.assertEqual(
            self.policy["unique_top1_tie_break"],
            ["p_final_desc", "q_market_desc", "entry_id_asc"],
        )
        self.assertEqual(self.policy["probability_tolerance"], 1e-6)
        self.assertEqual(self.policy["inference"]["bootstrap_unit"], "race_id")
        self.assertEqual(self.policy["inference"]["bootstrap_repetitions"], 10000)


if __name__ == "__main__":
    unittest.main()
