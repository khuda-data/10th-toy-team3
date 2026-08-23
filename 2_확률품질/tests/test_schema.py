from __future__ import annotations

import json
import unittest

from src.data.load_raw import DEFAULT_RAW_PATH, PROJECT_ROOT, load_raw
from src.data.validate_schema import (
    DEFAULT_MANIFEST_PATH,
    REQUIRED_COLUMNS,
    sha256_decompressed_gzip,
    sha256_file,
    validate_frame,
    validate_raw_file,
)


class RawSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_raw()
        cls.manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_canonical_raw_file_exists(self) -> None:
        self.assertTrue(DEFAULT_RAW_PATH.is_file())

    def test_required_columns_exist(self) -> None:
        self.assertTrue(REQUIRED_COLUMNS.issubset(self.frame.columns))

    def test_expected_shape(self) -> None:
        self.assertEqual(self.frame.shape, (56_648, 156))

    def test_entry_id_is_complete_and_unique(self) -> None:
        self.assertFalse(self.frame["entry_id"].isna().any())
        self.assertTrue(self.frame["entry_id"].is_unique)

    def test_identifier_columns_are_strings(self) -> None:
        for column in ["entry_id", "race_id", "hrNo", "jkNo", "trNo", "owNo"]:
            self.assertEqual(str(self.frame[column].dtype), "string")
        self.assertTrue((self.frame["hrNo"].str.len() == 7).all())
        self.assertGreater(self.frame["hrNo"].str.startswith("0").mean(), 0.9)

    def test_market_counts(self) -> None:
        counts = self.frame["meet"].value_counts().to_dict()
        self.assertEqual(counts, {"서울": 32_888, "부경": 23_760})

    def test_date_and_race_counts(self) -> None:
        self.assertEqual(int(self.frame["rcDate"].min()), 20_230_805)
        self.assertEqual(int(self.frame["rcDate"].max()), 20_260_809)
        self.assertEqual(self.frame["race_id"].nunique(), 5_361)

    def test_frame_validation_has_no_errors(self) -> None:
        result = validate_frame(self.frame)
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["warnings"]), 2)

    def test_hashes_match_manifest(self) -> None:
        self.assertEqual(
            sha256_file(DEFAULT_RAW_PATH),
            self.manifest["file"]["sha256"],
        )
        self.assertEqual(
            sha256_decompressed_gzip(DEFAULT_RAW_PATH),
            self.manifest["file"]["decompressed_sha256"],
        )

    def test_full_raw_validation_passes(self) -> None:
        result = validate_raw_file()
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["errors"], [])


if __name__ == "__main__":
    unittest.main()
