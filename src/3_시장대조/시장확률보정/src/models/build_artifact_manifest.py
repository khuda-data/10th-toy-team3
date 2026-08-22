"""Create checksums for completed stage 8-11 model artifacts and reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file


OUTPUT = PROJECT_ROOT / "data" / "manifests" / "model_baselines_manifest.json"
FILES = (
    "reports/experiments/m0_market_baseline.json",
    "reports/experiments/m1_logistic.json",
    "reports/experiments/m2_xgboost.json",
    "reports/experiments/stages_8_11_summary.md",
    "artifacts/models/m1_logistic.joblib",
    "artifacts/models/m2_xgboost.joblib",
    "data/predictions/m1_logistic_calibration.csv.gz",
    "data/predictions/m2_xgboost_calibration.csv.gz",
)


def main() -> int:
    artifacts = []
    for relative in FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development stages 8-11",
        "final_test_evaluated": False,
        "artifacts": artifacts,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
