from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads(
    (PROJECT_ROOT / "expected" / "metrics.json").read_text(encoding="utf-8")
)
TOLERANCE = float(EXPECTED["float_absolute_tolerance"])


class ReproducedMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage15 = json.loads(
            (PROJECT_ROOT / "reports" / "experiments" / "stage_15_final_test.json").read_text(
                encoding="utf-8"
            )
        )
        cls.stage26 = json.loads(
            (PROJECT_ROOT / "reports" / "experiments" / "stage_26_market_gate.json").read_text(
                encoding="utf-8"
            )
        )
        cls.stage27 = json.loads(
            (
                PROJECT_ROOT
                / "reports"
                / "experiments"
                / "stage_27_candidate_freeze.json"
            ).read_text(encoding="utf-8")
        )

    def assert_close(self, actual: float, expected: float) -> None:
        self.assertAlmostEqual(actual, expected, delta=TOLERANCE)

    def test_stage_15_final_test_metrics(self) -> None:
        expected = EXPECTED["stage_15"]
        self.assertEqual(self.stage15["test_fold"]["rows"], expected["test_rows"])
        self.assertEqual(self.stage15["test_fold"]["races"], expected["test_races"])
        market = self.stage15["market_baseline"]
        model = self.stage15["models"]["M2_xgboost"]["metrics"]["temperature_scaled_final"]
        self.assert_close(market["race_log_loss"], expected["market_race_log_loss"])
        self.assert_close(market["race_brier"], expected["market_race_brier"])
        self.assert_close(market["top1_accuracy"], expected["market_top1_accuracy"])
        self.assert_close(model["race_log_loss"], expected["m2_race_log_loss"])
        self.assert_close(model["race_brier"], expected["m2_race_brier"])
        self.assert_close(model["top1_accuracy"], expected["m2_top1_accuracy"])

    def test_stage_26_gate_metrics(self) -> None:
        expected = EXPECTED["stage_26"]
        selected = self.stage26["selected_candidate"]
        self.assertEqual(self.stage26["gate_calibration"]["races"], expected["calibration_races"])
        self.assertEqual(selected["top1_correct"], expected["top1_correct"])
        for key in ("threshold", "lambda_switch", "race_log_loss", "race_brier", "delta_logloss", "delta_brier"):
            self.assert_close(selected[key], expected[key])

    def test_stage_27_frozen_challenger(self) -> None:
        expected = EXPECTED["stage_27"]
        selected = self.stage27["selected_challenger"]
        self.assertEqual(self.stage27["challenger_id"], expected["challenger_id"])
        self.assertEqual(selected["candidate"], expected["selected_candidate"])
        self.assertEqual(self.stage27["status"], expected["status"])
        self.assertEqual(self.stage27["frozen_parameters"]["random_seed"], expected["random_seed"])
        self.assertEqual(selected["top1_correct"], expected["top1_correct"])
        for key in ("top1_accuracy", "race_log_loss", "race_brier"):
            self.assert_close(selected[key], expected[key])


if __name__ == "__main__":
    unittest.main()

