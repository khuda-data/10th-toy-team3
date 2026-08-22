from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file
from src.models.freeze_top1_candidate import (
    evaluate_candidate,
    select_constrained_candidate,
)


class Top1CandidateFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2", "r2"],
                "entry_id": ["a", "b", "c", "d"],
                "win": [1, 0, 0, 1],
                "q_market": [0.6, 0.4, 0.7, 0.3],
            }
        )

    def test_unique_top1_tie_break_is_used(self) -> None:
        result = evaluate_candidate(
            self.frame,
            np.array([0.5, 0.5, 0.5, 0.5]),
            name="R0_market",
        )
        self.assertEqual(result["top1_correct"], 1)
        self.assertEqual(result["actual_top1_override_races"], 0)

    def test_guardrails_are_applied_before_top1(self) -> None:
        candidates = [
            {
                "candidate": "bad_high_top1",
                "eligible_probability_guardrail": False,
                "top1_correct": 10,
                "actual_top1_override_races": 1,
                "complexity_rank": 1,
            },
            {
                "candidate": "safe",
                "eligible_probability_guardrail": True,
                "top1_correct": 9,
                "actual_top1_override_races": 2,
                "complexity_rank": 2,
            },
        ]
        self.assertEqual(select_constrained_candidate(candidates)["candidate"], "safe")

    def test_stage_27_freeze_contract_and_metrics(self) -> None:
        policy = json.loads(
            (PROJECT_ROOT / "data" / "manifests" / "top1_challenger_freeze.json").read_text(
                encoding="utf-8"
            )
        )
        report = json.loads(
            (PROJECT_ROOT / "reports" / "experiments" / "stage_27_candidate_freeze.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(policy["stage"], 27)
        self.assertEqual(policy["selected_candidate"], "R4_gated_adaptive_blend")
        self.assertEqual(policy["status"], "frozen_pending_future_holdout")
        self.assertFalse(policy["opened_final_test_evaluated"])
        self.assertEqual(report["data_policy"]["opened_final_test"], "not_loaded_not_evaluated")
        self.assertLessEqual(report["data_policy"]["date_max"], 20251227)
        self.assertEqual(policy["selected_metrics"]["top1_correct"], 249)
        self.assertEqual(policy["selected_metrics"]["delta_top1_correct"], 2)
        self.assertGreaterEqual(policy["selected_metrics"]["delta_logloss"], 0)
        self.assertGreaterEqual(policy["selected_metrics"]["delta_brier"], 0)
        self.assertEqual(policy["future_holdout"]["target_eligible_races"], 500)
        self.assertTrue(policy["future_holdout"]["single_open_after_target"])

        by_name = {row["candidate"]: row for row in report["candidates"]}
        self.assertTrue(by_name["R3_fixed_m2_market_blend"]["eligible_probability_guardrail"])
        self.assertFalse(by_name["R2_ranker_probability"]["eligible_probability_guardrail"])
        for item in policy["frozen_components"] + policy["outputs"]:
            self.assertEqual(sha256_file(PROJECT_ROOT / item["path"]), item["sha256"])

    def test_future_validation_protocol_is_preserved(self) -> None:
        protocol = PROJECT_ROOT / "FUTURE_HOLDOUT_VALIDATION.md"
        text = protocol.read_text(encoding="utf-8")
        for required in (
            "첫 500개 적격 서울 경주",
            "paired bootstrap 10,000회",
            "exact McNemar",
            "중간 Top-1",
            "Legacy Test",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
