import unittest
import json
from pathlib import Path

from src.data.load_raw import PROJECT_ROOT


class DocumentationTests(unittest.TestCase):
    def test_required_project_guides_exist_and_are_not_placeholders(self):
        for name in (
            "README.md",
            "SETUP.md",
            "TESTING.md",
            "PROJECT_GUIDELINES.md",
            "FUTURE_HOLDOUT_VALIDATION.md",
        ):
            path = PROJECT_ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(len(path.read_text(encoding="utf-8")), 500, name)

    def test_readme_records_locked_result_and_no_bet_policy(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for required in (
            "Race Log Loss",
            "Race Brier",
            "m2_xgboost_sum_l005_t095_v1",
            "no_bet",
            "Final Test",
            "python -m unittest discover -s tests -v",
        ):
            self.assertIn(required, readme)

    def test_documented_result_files_exist(self):
        expected = [
            "reports/experiments/stages_8_11_summary.md",
            *[f"reports/experiments/stage_{stage}_summary.md" for stage in range(12, 20)],
            "data/manifests/prediction_output_schema.json",
            "data/predictions/stage_18_contract_fixture.csv",
            "scripts/run_tests.ps1",
        ]
        for relative in expected:
            self.assertTrue((PROJECT_ROOT / Path(relative)).is_file(), relative)

    def test_documentation_manifest_matches_current_files(self):
        from src.data.validate_schema import sha256_file

        path = PROJECT_ROOT / "data" / "manifests" / "documentation_manifest.json"
        self.assertTrue(path.is_file())
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["test_framework"], "unittest discovery")
        self.assertEqual(manifest["test_module_count"], 18)
        for item in manifest["files"]:
            self.assertEqual(sha256_file(PROJECT_ROOT / item["path"]), item["sha256"])


if __name__ == "__main__":
    unittest.main()
