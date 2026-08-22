import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file


class ExperimentReportTests(unittest.TestCase):
    def test_completed_baseline_reports_keep_test_sealed(self):
        for name in (
            "m0_market_baseline.json",
            "m1_logistic.json",
            "m2_xgboost.json",
        ):
            path = PROJECT_ROOT / "reports" / "experiments" / name
            self.assertTrue(path.is_file(), name)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(report["test_policy"]["evaluated"])
            self.assertNotIn("test", report["evaluation"])

    def test_model_artifacts_and_calibration_predictions_exist(self):
        expected = (
            "artifacts/models/m1_logistic.joblib",
            "artifacts/models/m2_xgboost.joblib",
            "data/predictions/m1_logistic_calibration.csv.gz",
            "data/predictions/m2_xgboost_calibration.csv.gz",
        )
        for relative in expected:
            path = PROJECT_ROOT / Path(relative)
            self.assertTrue(path.is_file(), relative)
            self.assertGreater(path.stat().st_size, 0, relative)

    def test_artifact_manifest_keeps_final_test_sealed(self):
        path = (
            PROJECT_ROOT / "data" / "manifests" / "model_baselines_manifest.json"
        )
        self.assertTrue(path.is_file())
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["final_test_evaluated"])
        self.assertEqual(len(manifest["artifacts"]), 8)

    def test_normalization_policy_was_selected_without_final_test(self):
        policy_path = (
            PROJECT_ROOT / "data" / "manifests" / "normalization_policy.json"
        )
        report_path = (
            PROJECT_ROOT / "reports" / "experiments" / "stage_12_normalization.json"
        )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(policy["final_test_evaluated"])
        self.assertFalse(report["test_policy"]["evaluated"])
        self.assertEqual(
            policy["selected_methods"],
            {"M1_logistic": "sum", "M2_xgboost": "sum"},
        )
        for details in report["models"].values():
            self.assertEqual(details["selection_data"], "Train walk-forward OOF only")
            self.assertEqual(details["oof_races"], 770)

    def test_selected_calibration_predictions_form_race_simplexes(self):
        for name in (
            "m1_logistic_calibration_normalized.csv.gz",
            "m2_xgboost_calibration_normalized.csv.gz",
        ):
            frame = pd.read_csv(PROJECT_ROOT / "data" / "predictions" / name)
            self.assertEqual(len(frame), 6582)
            self.assertEqual(frame["race_id"].nunique(), 641)
            self.assertTrue(frame["normalization_method"].eq("sum").all())
            sums = frame.groupby("race_id")["p_model_race"].sum().to_numpy()
            self.assertTrue(np.allclose(sums, 1.0, atol=1e-9))

    def test_market_blend_policy_uses_calibration_and_keeps_test_sealed(self):
        policy_path = (
            PROJECT_ROOT / "data" / "manifests" / "market_blend_policy.json"
        )
        report_path = (
            PROJECT_ROOT / "reports" / "experiments" / "stage_13_market_blend.json"
        )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(policy["selection_fold"], "calibration")
        self.assertFalse(policy["final_test_evaluated"])
        self.assertFalse(report["test_policy"]["evaluated"])
        self.assertEqual(
            policy["selected_lambdas"],
            {"M1_logistic": 0.05, "M2_xgboost": 0.05},
        )
        self.assertEqual(policy["deployment_candidate"]["model"], "M2_xgboost")
        for details in report["models"].values():
            self.assertEqual(len(details["lambda_grid"]), 10)
            self.assertEqual(details["races"], 641)

    def test_blended_calibration_predictions_form_race_simplexes(self):
        for name in (
            "m1_logistic_calibration_blended.csv.gz",
            "m2_xgboost_calibration_blended.csv.gz",
        ):
            frame = pd.read_csv(PROJECT_ROOT / "data" / "predictions" / name)
            self.assertEqual(len(frame), 6582)
            self.assertEqual(frame["race_id"].nunique(), 641)
            self.assertTrue(frame["blend_lambda"].eq(0.05).all())
            sums = frame.groupby("race_id")["p_blend"].sum().to_numpy()
            self.assertTrue(np.allclose(sums, 1.0, atol=1e-9))

    def test_temperature_policy_freezes_sequential_calibration(self):
        policy_path = (
            PROJECT_ROOT / "data" / "manifests" / "temperature_policy.json"
        )
        report_path = (
            PROJECT_ROOT
            / "reports"
            / "experiments"
            / "stage_14_temperature_scaling.json"
        )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(policy["selection_fold"], "calibration")
        self.assertFalse(policy["final_test_evaluated"])
        self.assertFalse(report["test_policy"]["evaluated"])
        self.assertEqual(
            policy["selected_temperatures"],
            {"M1_logistic": 0.95, "M2_xgboost": 0.95},
        )
        self.assertEqual(policy["deployment_candidate"]["model"], "M2_xgboost")
        self.assertIn("optimistic", policy["sequential_selection_warning"])
        for details in report["models"].values():
            self.assertEqual(len(details["temperature_grid"]), 10)
            self.assertEqual(details["blend_lambda"], 0.05)
            self.assertEqual(details["races"], 641)

    def test_temperature_scaled_predictions_form_race_simplexes(self):
        for name in (
            "m1_logistic_calibration_final.csv.gz",
            "m2_xgboost_calibration_final.csv.gz",
        ):
            frame = pd.read_csv(PROJECT_ROOT / "data" / "predictions" / name)
            self.assertEqual(len(frame), 6582)
            self.assertEqual(frame["race_id"].nunique(), 641)
            self.assertTrue(frame["blend_lambda"].eq(0.05).all())
            self.assertTrue(frame["temperature"].eq(0.95).all())
            sums = frame.groupby("race_id")["p_final"].sum().to_numpy()
            self.assertTrue(np.allclose(sums, 1.0, atol=1e-9))

    def test_final_test_was_frozen_and_evaluated_exactly_once(self):
        freeze_path = (
            PROJECT_ROOT / "data" / "manifests" / "pre_final_test_freeze.json"
        )
        result_path = (
            PROJECT_ROOT / "data" / "manifests" / "final_test_evaluation.json"
        )
        report_path = (
            PROJECT_ROOT / "reports" / "experiments" / "stage_15_final_test.json"
        )
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(len(freeze["files"]), 19)
        self.assertEqual(result["evaluation_count"], 1)
        self.assertTrue(result["final_test_evaluated"])
        self.assertTrue(report["test_policy"]["evaluated"])
        self.assertEqual(report["test_policy"]["evaluation_count"], 1)
        self.assertEqual(
            freeze["frozen_deployment_candidate"],
            {"model": "M2_xgboost", "normalization": "sum", "lambda": 0.05, "temperature": 0.95},
        )
        self.assertTrue(
            report["primary_conclusion"]["beats_market_on_both_primary_metrics"]
        )
        for item in freeze["files"]:
            self.assertEqual(sha256_file(PROJECT_ROOT / item["path"]), item["sha256"])
        for item in result["outputs"]:
            self.assertEqual(sha256_file(PROJECT_ROOT / item["path"]), item["sha256"])

    def test_final_test_predictions_form_race_simplexes(self):
        for name in (
            "m1_logistic_test_final.csv.gz",
            "m2_xgboost_test_final.csv.gz",
        ):
            frame = pd.read_csv(PROJECT_ROOT / "data" / "predictions" / name)
            self.assertEqual(len(frame), 6639)
            self.assertEqual(frame["race_id"].nunique(), 635)
            self.assertTrue(frame["blend_lambda"].eq(0.05).all())
            self.assertTrue(frame["temperature"].eq(0.95).all())
            for column in ("q_market", "p_model_race", "p_blend", "p_final"):
                sums = frame.groupby("race_id")[column].sum().to_numpy()
                self.assertTrue(np.allclose(sums, 1.0, atol=1e-9), column)

    def test_bootstrap_report_uses_locked_paired_race_resampling(self):
        report_path = (
            PROJECT_ROOT / "reports" / "experiments" / "stage_16_bootstrap.json"
        )
        manifest_path = (
            PROJECT_ROOT / "data" / "manifests" / "bootstrap_evaluation.json"
        )
        final_path = (
            PROJECT_ROOT / "data" / "manifests" / "final_test_evaluation.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(report["method"]["unit"], "race_id")
        self.assertTrue(report["method"]["paired"])
        self.assertEqual(report["method"]["n_bootstrap"], 5000)
        self.assertEqual(report["method"]["random_seed"], 42)
        self.assertEqual(
            manifest["input_final_test_manifest_sha256"], sha256_file(final_path)
        )
        self.assertFalse(manifest["final_test_model_changed"])
        conclusion = report["primary_conclusion"]
        self.assertTrue(conclusion["race_log_loss_ci_lower_above_zero"])
        self.assertFalse(conclusion["race_brier_ci_lower_above_zero"])
        self.assertFalse(conclusion["statistically_supported_on_both_primary_metrics"])
        for item in manifest["outputs"]:
            self.assertEqual(sha256_file(PROJECT_ROOT / item["path"]), item["sha256"])

    def test_bootstrap_replicates_are_complete_and_finite(self):
        path = (
            PROJECT_ROOT
            / "data"
            / "analysis"
            / "stage_16_bootstrap_replicates.csv.gz"
        )
        frame = pd.read_csv(path)
        self.assertEqual(len(frame), 5000)
        self.assertEqual(frame["replicate"].tolist(), list(range(1, 5001)))
        self.assertEqual(len(frame.columns), 7)
        self.assertTrue(np.isfinite(frame.to_numpy()).all())

    def test_backtest_policy_is_calibration_selected_no_bet(self):
        policy_path = PROJECT_ROOT / "data" / "manifests" / "betting_policy.json"
        report_path = (
            PROJECT_ROOT / "reports" / "experiments" / "stage_17_backtest.json"
        )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(policy["selection_fold"], "calibration")
        self.assertEqual(policy["deployment_policy"]["action"], "no_bet")
        self.assertIsNone(policy["deployment_policy"]["threshold"])
        self.assertTrue(policy["uses_closing_odds"])
        self.assertFalse(policy["executable_live_strategy"])
        self.assertFalse(policy["final_test_model_changed"])
        self.assertEqual(report["final_policy"]["action"], "no_bet")
        for fold in ("calibration", "test_descriptive_only"):
            for row in report[fold]["threshold_results"]:
                self.assertEqual(row["bets"], 0)
                self.assertIsNone(row["roi"])
                self.assertEqual(row["roi_bootstrap"]["valid_replicates"], 0)
        for item in policy["outputs"]:
            self.assertEqual(sha256_file(PROJECT_ROOT / item["path"]), item["sha256"])

    def test_backtest_selection_file_is_valid_empty_contract(self):
        path = (
            PROJECT_ROOT / "data" / "analysis" / "stage_17_bet_selections.csv.gz"
        )
        frame = pd.read_csv(path)
        self.assertTrue(frame.empty)
        self.assertEqual(
            frame.columns.tolist(),
            [
                "fold",
                "threshold",
                "race_id",
                "entry_id",
                "rcDate",
                "win",
                "winOdds",
                "q_market",
                "p_final",
                "break_even_prob",
                "expected_edge",
                "profit",
            ],
        )

    def test_prediction_output_schema_and_fixture_contract(self):
        schema_path = (
            PROJECT_ROOT / "data" / "manifests" / "prediction_output_schema.json"
        )
        fixture_path = (
            PROJECT_ROOT / "data" / "predictions" / "stage_18_contract_fixture.csv"
        )
        report_path = (
            PROJECT_ROOT
            / "reports"
            / "experiments"
            / "stage_18_prediction_contract.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fixture = pd.read_csv(fixture_path, encoding="utf-8-sig", dtype={"hrNo": "string"})
        self.assertEqual(fixture.columns.tolist(), schema["columns"])
        self.assertEqual(len(fixture), 21)
        self.assertEqual(fixture["race_id"].nunique(), 2)
        self.assertFalse(fixture["entry_id"].duplicated().any())
        self.assertTrue(fixture["action"].eq("no_bet").all())
        self.assertEqual(fixture["model_version"].nunique(), 1)
        self.assertEqual(
            fixture["model_version"].iloc[0], "m2_xgboost_sum_l005_t095_v1"
        )
        for forbidden in schema["contracts"]["result_columns_forbidden"]:
            self.assertNotIn(forbidden, fixture.columns)
        for column in ("q_market", "p_premarket", "p_final"):
            sums = fixture.groupby("race_id")[column].sum().to_numpy()
            self.assertTrue(np.allclose(sums, 1.0, atol=1e-6), column)
        np.testing.assert_allclose(
            fixture["market_delta"], fixture["p_final"] - fixture["q_market"]
        )
        np.testing.assert_allclose(
            fixture["break_even_prob"], 1.0 / fixture["winOdds_snapshot"]
        )
        np.testing.assert_allclose(
            fixture["expected_edge"],
            fixture["p_final"] * fixture["winOdds_snapshot"] - 1.0,
        )
        snapshot = pd.to_datetime(fixture["odds_snapshot_time"], utc=True)
        prediction = pd.to_datetime(fixture["prediction_time"], utc=True)
        start = pd.to_datetime(fixture["race_start_time"], utc=True)
        self.assertTrue((snapshot <= prediction).all())
        self.assertTrue((prediction < start).all())
        self.assertEqual(report["fixture"]["sha256"], sha256_file(fixture_path))
        self.assertFalse(report["live_readiness"])
        self.assertFalse(report["final_test_model_changed"])


if __name__ == "__main__":
    unittest.main()
