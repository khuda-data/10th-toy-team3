from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pandas
import sklearn
import xgboost


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.config import BOOTSTRAP_REPS, CONFIG_DIR, PERCENTILES, RANDOM_SEED  # noqa: E402


def main() -> None:
    draft_path = CONFIG_DIR / "selection_draft.json"
    if not draft_path.exists():
        raise FileNotFoundError("Run 02_train_validate.py first")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    locked = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "test_policy": "single_evaluation_no_retuning",
        "primary_feature_set": draft["primary_feature_set"],
        "percentiles": list(PERCENTILES),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_unit": "race_id",
        "random_seed": RANDOM_SEED,
        "selections": draft["selections"],
        "environment": {
            "python": platform.python_version(),
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    canonical = json.dumps(locked, ensure_ascii=False, sort_keys=True).encode("utf-8")
    locked["sha256"] = hashlib.sha256(canonical).hexdigest()
    output = CONFIG_DIR / "locked_config.json"
    output.write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[LOCKED] {output}")
    print(f"sha256={locked['sha256']}")


if __name__ == "__main__":
    main()
