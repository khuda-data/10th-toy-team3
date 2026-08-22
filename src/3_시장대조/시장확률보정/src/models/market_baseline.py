"""Evaluate and freeze M0, the normalized market-probability baseline."""

from __future__ import annotations

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries
from src.evaluation.race_metrics import race_metrics
from src.models.common import REPORT_DIR, save_report, utc_now


def main() -> int:
    results = {}
    for fold in ("train", "calibration"):
        frame = load_model_entries((fold,))
        results[fold] = race_metrics(frame, frame["q_market"].to_numpy())
    report = {
        "experiment": "M0_market",
        "created_at": utc_now(),
        "probability": "q_market = inverse win odds normalized within race",
        "evaluation": results,
        "test_policy": {
            "evaluated": False,
            "reason": "Final test is sealed until model and calibration policy are frozen.",
        },
    }
    save_report(REPORT_DIR / "m0_market_baseline.json", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
