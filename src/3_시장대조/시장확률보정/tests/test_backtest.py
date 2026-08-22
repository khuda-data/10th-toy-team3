import unittest

import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    equal_stake_metrics,
    roi_bootstrap,
    selected_bets,
    select_calibration_policy,
)


class BacktestUnitTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "race_id": ["a", "a", "b", "b"],
                "rcDate": [1, 1, 2, 2],
                "entry_id": ["a1", "a2", "b1", "b2"],
                "win": [1, 0, 0, 1],
                "winOdds": [3.0, 4.0, 2.0, 5.0],
                "p_final": [0.5, 0.2, 0.6, 0.3],
                "expected_edge": [0.5, -0.2, 0.2, 0.5],
            }
        )

    def test_equal_stake_profit_uses_decimal_odds(self):
        selected = selected_bets(self.frame, 0.4)
        np.testing.assert_allclose(selected["profit"], [2.0, 4.0])
        metrics = equal_stake_metrics(self.frame, 0.4)
        self.assertEqual(metrics["bets"], 2)
        self.assertEqual(metrics["wins"], 2)
        self.assertEqual(metrics["net_profit"], 6.0)
        self.assertEqual(metrics["roi"], 3.0)

    def test_policy_falls_back_to_no_bet_without_confident_candidate(self):
        rows = [
            {
                "threshold": 0.05,
                "bets": 100,
                "roi": 0.1,
                "roi_bootstrap": {
                    "ci_lower_above_zero": False,
                    "ci_95_percentile": {"lower": -0.1},
                },
            }
        ]
        policy = select_calibration_policy(rows)
        self.assertEqual(policy["deployment_policy"]["action"], "no_bet")

    def test_empty_selection_has_safe_null_roi_interval(self):
        result = roi_bootstrap(self.frame, 10.0, seed=42)
        self.assertEqual(result["valid_replicates"], 0)
        self.assertIsNone(result["ci_95_percentile"]["lower"])
        self.assertFalse(result["ci_lower_above_zero"])


if __name__ == "__main__":
    unittest.main()
