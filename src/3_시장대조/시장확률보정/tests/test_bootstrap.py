import unittest

import numpy as np

from src.evaluation.bootstrap import bootstrap_means, summarize_delta


class BootstrapUnitTests(unittest.TestCase):
    def test_bootstrap_means_use_paired_sample_indices(self):
        values = np.array([1.0, 2.0, 3.0])
        indices = np.array([[0, 0, 0], [0, 1, 2], [2, 2, 2]])
        np.testing.assert_allclose(bootstrap_means(values, indices), [1.0, 2.0, 3.0])

    def test_summary_reports_probability_and_positive_lower_bound(self):
        replicates = np.linspace(0.1, 0.2, 1000)
        summary = summarize_delta(0.15, replicates)
        self.assertEqual(summary["probability_model_better"], 1.0)
        self.assertTrue(summary["ci_lower_above_zero"])
        self.assertGreater(summary["ci_95_percentile"]["lower"], 0.0)


if __name__ == "__main__":
    unittest.main()
