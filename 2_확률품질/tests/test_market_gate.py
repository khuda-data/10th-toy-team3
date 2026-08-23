from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file
from src.models.market_gate import adaptive_geometric_blend, build_gate_race_table


class MarketGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2", "r2"],
                "entry_id": ["a", "b", "c", "d"],
                "rcDate": [20250101, 20250101, 20250102, 20250102],
                "win": [0, 1, 1, 0],
                "q_market": [0.7, 0.3, 0.6, 0.4],
                "ranking_score": [0.1, 0.9, 0.8, 0.2],
                "p_ranker_race": [0.3, 0.7, 0.7, 0.3],
                "age": [3, 4, 4, 3],
                "rcDist": [1200] * 4,
                "waterRate": [5.0] * 4,
                "rank": ["A"] * 4,
                "track": ["dry"] * 4,
                "weather": ["sunny"] * 4,
            }
        )

    def test_gate_labels_beneficial_and_agreement(self) -> None:
        races = build_gate_race_table(self.frame, feature_columns=("age",)).set_index("race_id")
        self.assertTrue(races.loc["r1", "top1_disagreement"])
        self.assertTrue(races.loc["r1", "switch_beneficial"])
        self.assertFalse(races.loc["r2", "top1_disagreement"])
        self.assertFalse(races.loc["r2", "switch_beneficial"])

    def test_adaptive_blend_endpoints_and_simplex(self) -> None:
        market = self.frame["q_market"].to_numpy()
        ranker = self.frame["p_ranker_race"].to_numpy()
        probabilities = adaptive_geometric_blend(
            self.frame,
            market,
            ranker,
            {"r1": 0.0, "r2": 1.0},
        )
        np.testing.assert_allclose(probabilities[:2], market[:2])
        np.testing.assert_allclose(probabilities[2:], ranker[2:])
        sums = pd.Series(probabilities).groupby(self.frame["race_id"]).sum()
        np.testing.assert_allclose(sums, 1.0)

    def test_invalid_lambda_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            adaptive_geometric_blend(
                self.frame,
                self.frame["q_market"],
                self.frame["p_ranker_race"],
                {"r1": 0.0, "r2": 1.1},
            )

    def test_stage_26_policy_and_outputs(self) -> None:
        policy = json.loads(
            (PROJECT_ROOT / "data" / "manifests" / "market_gate_policy.json").read_text(
                encoding="utf-8"
            )
        )
        report = json.loads(
            (PROJECT_ROOT / "reports" / "experiments" / "stage_26_market_gate.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["data_policy"]["gate_fit"], "Train chronological OOF disagreement races only")
        self.assertEqual(report["data_policy"]["opened_final_test"], "not_loaded_not_evaluated")
        self.assertLessEqual(report["data_policy"]["max_rcDate"], 20251227)
        self.assertFalse(policy["opened_final_test_evaluated"])
        self.assertGreaterEqual(report["selected_candidate"]["delta_logloss"], -1e-12)
        self.assertGreaterEqual(report["selected_candidate"]["delta_brier"], -1e-12)
        self.assertEqual(report["selection_grid"]["candidate_count_including_market_baseline"], 109)
        self.assertEqual(policy["selected_threshold"], 0.65)
        self.assertEqual(policy["selected_lambda_switch"], 0.3)
        self.assertEqual(report["selected_candidate"]["top1_correct"], 249)
        self.assertEqual(report["selected_candidate"]["actual_top1_override_races"], 14)
        self.assertEqual(report["accuracy_only_best"]["top1_correct"], 252)
        self.assertFalse(report["accuracy_only_best"]["eligible_probability_guardrail"])
        for item in policy["outputs"]:
            self.assertEqual(sha256_file(PROJECT_ROOT / item["path"]), item["sha256"])
        predictions = pd.read_csv(
            PROJECT_ROOT / "data" / "predictions" / "r4_gated_calibration.csv.gz"
        )
        sums = predictions.groupby("race_id")["p_final"].sum().to_numpy()
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
