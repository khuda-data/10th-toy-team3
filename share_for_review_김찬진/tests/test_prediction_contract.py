import unittest

import pandas as pd

from src.inference.prediction_contract import generate_predictions, race_rejection_reasons


class PredictionContractUnitTests(unittest.TestCase):
    def base_frame(self):
        return pd.DataFrame(
            {
                "race_id": ["r1", "r1"],
                "entry_id": ["e1", "e2"],
                "hrNo": ["0000001", "0000002"],
                "hrName": ["A", "B"],
                "dusu": [2, 2],
                "winOdds_snapshot": [2.0, 3.0],
                "odds_snapshot_time": ["2026-01-01T11:55:00+09:00"] * 2,
                "prediction_time": ["2026-01-01T11:56:00+09:00"] * 2,
                "race_start_time": ["2026-01-01T12:00:00+09:00"] * 2,
                "odds_source": ["unit_test"] * 2,
            }
        )

    def test_complete_race_is_accepted(self):
        self.assertEqual(race_rejection_reasons(self.base_frame()), {})

    def test_incomplete_race_is_rejected_whole(self):
        frame = self.base_frame().iloc[:1].copy()
        self.assertEqual(
            race_rejection_reasons(frame), {"r1": "incomplete_race_entries"}
        )
        output = generate_predictions(frame)
        self.assertTrue(output["action"].eq("prediction_rejected").all())
        self.assertTrue(output["p_final"].isna().all())
        self.assertTrue(output["rejection_reason"].eq("incomplete_race_entries").all())

    def test_future_snapshot_or_late_prediction_is_rejected(self):
        frame = self.base_frame()
        frame["prediction_time"] = "2026-01-01T12:01:00+09:00"
        self.assertEqual(race_rejection_reasons(frame), {"r1": "invalid_time_order"})


if __name__ == "__main__":
    unittest.main()
