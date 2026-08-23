import unittest

import numpy as np
import pandas as pd

from src.data.model_data import load_model_entries, make_walk_forward_folds
from src.evaluation.race_metrics import (
    geometric_blend,
    normalize_by_race,
    normalize_model_probabilities,
    race_metrics,
    temperature_scale,
)
from src.features.preprocess import QuantileClipper, infer_feature_schema


class ModelingFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_model_entries(("train",))

    def test_model_loader_returns_only_complete_normal_races(self):
        self.assertEqual(len(self.train), 19617)
        self.assertEqual(self.train["race_id"].nunique(), 1891)
        self.assertTrue(self.train.groupby("race_id")["win"].sum().eq(1).all())

    def test_walk_forward_is_strictly_chronological(self):
        folds = make_walk_forward_folds(self.train)
        self.assertEqual(len(folds), 4)
        for fold in folds:
            self.assertLess(fold["train_date_max"], fold["valid_date_min"])

    def test_feature_schema_contains_only_reviewed_premarket_features(self):
        schema = infer_feature_schema(self.train)
        self.assertEqual(len(schema.features), 112)
        self.assertEqual(len(schema.numeric), 99)
        self.assertEqual(len(schema.categorical), 13)
        self.assertNotIn("q_market", schema.features)
        self.assertNotIn("win", schema.features)

    def test_quantile_clipper_does_not_refit_during_transform(self):
        clipper = QuantileClipper(lower=0.0, upper=0.75).fit([[0], [1], [2], [3]])
        before = clipper.upper_bounds_.copy()
        transformed = clipper.transform([[100]])
        np.testing.assert_array_equal(before, clipper.upper_bounds_)
        self.assertLess(transformed[0, 0], 100)

    def test_race_normalization_and_metrics(self):
        frame = pd.DataFrame(
            {"race_id": ["a", "a", "b", "b"], "win": [1, 0, 0, 1]}
        )
        p = normalize_by_race(frame, [2, 1, 1, 3])
        metrics = race_metrics(frame, p)
        self.assertTrue(np.allclose(pd.Series(p).groupby(frame["race_id"]).sum(), 1))
        self.assertEqual(metrics["races"], 2)
        self.assertEqual(metrics["top1_accuracy"], 1.0)

    def test_logit_softmax_is_a_valid_and_distinct_simplex(self):
        frame = pd.DataFrame(
            {"race_id": ["a", "a", "b", "b"], "win": [1, 0, 0, 1]}
        )
        raw = [0.2, 0.4, 0.3, 0.3]
        summed = normalize_model_probabilities(frame, raw, method="sum")
        softmax = normalize_model_probabilities(
            frame, raw, method="logit_softmax"
        )
        self.assertFalse(np.allclose(summed[:2], softmax[:2]))
        self.assertTrue(
            np.allclose(pd.Series(softmax).groupby(frame["race_id"]).sum(), 1)
        )

    def test_unknown_normalization_is_rejected(self):
        frame = pd.DataFrame({"race_id": ["a"], "win": [1]})
        with self.assertRaises(ValueError):
            normalize_model_probabilities(frame, [0.5], method="unknown")

    def test_geometric_blend_endpoints_match_market_and_model(self):
        frame = pd.DataFrame(
            {"race_id": ["a", "a", "b", "b"], "win": [1, 0, 0, 1]}
        )
        market = np.array([0.7, 0.3, 0.4, 0.6])
        model = np.array([0.2, 0.8, 0.9, 0.1])
        np.testing.assert_allclose(
            geometric_blend(frame, market, model, lam=0.0), market
        )
        np.testing.assert_allclose(
            geometric_blend(frame, market, model, lam=1.0), model
        )

    def test_geometric_blend_rejects_lambda_outside_unit_interval(self):
        frame = pd.DataFrame({"race_id": ["a"], "win": [1]})
        with self.assertRaises(ValueError):
            geometric_blend(frame, [1.0], [1.0], lam=1.01)

    def test_temperature_one_is_identity_and_preserves_simplex(self):
        frame = pd.DataFrame(
            {"race_id": ["a", "a", "b", "b"], "win": [1, 0, 0, 1]}
        )
        probabilities = np.array([0.7, 0.3, 0.4, 0.6])
        scaled = temperature_scale(frame, probabilities, temperature=1.0)
        np.testing.assert_allclose(scaled, probabilities)
        self.assertTrue(
            np.allclose(pd.Series(scaled).groupby(frame["race_id"]).sum(), 1.0)
        )

    def test_temperature_controls_sharpness_and_rejects_nonpositive_values(self):
        frame = pd.DataFrame({"race_id": ["a", "a"], "win": [1, 0]})
        probabilities = np.array([0.7, 0.3])
        sharp = temperature_scale(frame, probabilities, temperature=0.5)
        flat = temperature_scale(frame, probabilities, temperature=2.0)
        self.assertGreater(sharp.max(), probabilities.max())
        self.assertLess(flat.max(), probabilities.max())
        with self.assertRaises(ValueError):
            temperature_scale(frame, probabilities, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
