"""Freeze all model, policy, data, and evaluation inputs before opening Final Test."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file


OUTPUT = PROJECT_ROOT / "data" / "manifests" / "pre_final_test_freeze.json"
FROZEN_FILES = (
    "artifacts/models/m1_logistic.joblib",
    "artifacts/models/m2_xgboost.joblib",
    "data/interim/seoul_entries.csv.gz",
    "data/interim/split_manifest.csv",
    "data/manifests/feature_registry.json",
    "data/manifests/seoul_interim_manifest.json",
    "data/manifests/split_manifest.json",
    "data/manifests/normalization_policy.json",
    "data/manifests/market_blend_policy.json",
    "data/manifests/temperature_policy.json",
    "reports/experiments/m1_logistic.json",
    "reports/experiments/m2_xgboost.json",
    "reports/experiments/stage_12_normalization.json",
    "reports/experiments/stage_13_market_blend.json",
    "reports/experiments/stage_14_temperature_scaling.json",
    "src/data/model_data.py",
    "src/evaluation/race_metrics.py",
    "src/features/preprocess.py",
    "src/models/evaluate_final_test.py",
)


def load_json(relative: str) -> dict:
    return json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))


def frozen_candidate() -> dict:
    normalization = load_json("data/manifests/normalization_policy.json")
    blend = load_json("data/manifests/market_blend_policy.json")
    temperature = load_json("data/manifests/temperature_policy.json")
    candidate = temperature["deployment_candidate"]
    model = candidate["model"]
    if candidate["lambda"] != blend["selected_lambdas"][model]:
        raise ValueError("Temperature and blend policies disagree on lambda")
    if candidate["temperature"] != temperature["selected_temperatures"][model]:
        raise ValueError("Temperature policy disagrees with deployment candidate")
    return {
        "model": model,
        "normalization": normalization["selected_methods"][model],
        "lambda": candidate["lambda"],
        "temperature": candidate["temperature"],
    }


def build_manifest() -> dict:
    files = []
    for relative in FROZEN_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "freeze_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Immutable inputs for the single Final Test evaluation",
        "test_fold": {"rows": 6639, "races": 635, "date_min": 20251228, "date_max": 20260809},
        "frozen_deployment_candidate": frozen_candidate(),
        "files": files,
        "final_test_opened": False,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Freeze manifest already exists and will not be overwritten: {OUTPUT}"
        )
    manifest = build_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
