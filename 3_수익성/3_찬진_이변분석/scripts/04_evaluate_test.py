from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.config import (  # noqa: E402
    CONFIG_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    ensure_output_dirs,
)
from src.data import load_fold, subset_and_target, validate_stored_label  # noqa: E402
from src.features import assert_no_leakage, select_features  # noqa: E402
from src.metrics import PlattCalibrator, classification_metrics  # noqa: E402
from src.modeling import predict_scores  # noqa: E402
from src.roi import cluster_bootstrap_roi, percentile_roi_tables  # noqa: E402


def main() -> None:
    ensure_output_dirs()
    lock_path = CONFIG_DIR / "locked_config.json"
    if not lock_path.exists():
        raise FileNotFoundError("Configuration is not locked. Run 03_lock_config.py first.")
    locked = json.loads(lock_path.read_text(encoding="utf-8"))

    # This is the only point in the pipeline that opens test outcomes.
    test_all = load_fold("test", include_outcomes=True)
    metric_records: list[dict] = []
    audit: dict = {"test_rows": int(len(test_all)), "targets": {}, "odds": {}}
    operational_records: list[dict] = []

    for target_name, feature_sets in locked["selections"].items():
        test, y_test = subset_and_target(test_all, target_name)
        agreement = validate_stored_label(test_all, target_name)
        if agreement != 1.0:
            raise ValueError(f"test/{target_name}: stored-label agreement={agreement}")
        audit["targets"][target_name] = {
            "rows": int(len(test)),
            "positives": int(y_test.sum()),
            "base_rate": float(y_test.mean()),
            "stored_label_agreement": agreement,
        }
        for feature_set, selection in feature_sets.items():
            bundle = joblib.load(selection["bundle"])
            X_test = select_features(test, feature_set).reindex(columns=bundle["feature_columns"])
            assert_no_leakage(X_test, feature_set)
            test_matrix = bundle["preprocessor"].transform(X_test)
            scores = predict_scores(bundle["model"], test_matrix)

            calibrator = PlattCalibrator().fit(bundle["valid_scores"], bundle["valid_target"])
            calibrated = calibrator.predict(scores)
            metrics = classification_metrics(y_test.to_numpy(), scores)
            calibrated_metrics = classification_metrics(y_test.to_numpy(), calibrated)
            metric_records.append(
                {
                    "target": target_name,
                    "feature_set": feature_set,
                    "candidate": selection["candidate"],
                    **metrics,
                    "calibrated_brier": calibrated_metrics["brier"],
                }
            )

            predictions = pd.DataFrame(
                {
                    "entry_id": test["entry_id"].astype(str),
                    "race_id": test["race_id"].astype(str),
                    "rcDate": test["rcDate"],
                    "target": y_test.to_numpy(int),
                    "score": scores,
                    "calibrated_probability": calibrated,
                    "plcOdds": pd.to_numeric(test["plcOdds"], errors="coerce"),
                    "place": test["place"].astype(int),
                }
            )
            prediction_path = PREDICTION_DIR / f"test_{target_name}_{feature_set}.csv"
            predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

            if target_name == "darkhorse":
                valid_odds = predictions["plcOdds"].notna() & predictions["plcOdds"].gt(0) & predictions["plcOdds"].lt(999)
                audit["odds"][feature_set] = {
                    "rows": int(len(predictions)),
                    "valid_rows": int(valid_odds.sum()),
                    "missing_rows": int(predictions["plcOdds"].isna().sum()),
                    "nonpositive_rows": int(predictions["plcOdds"].le(0).sum()),
                    "sentinel_or_extreme_rows": int(predictions["plcOdds"].ge(999).sum()),
                    "min_valid": float(predictions.loc[valid_odds, "plcOdds"].min()),
                    "median_valid": float(predictions.loc[valid_odds, "plcOdds"].median()),
                    "max_valid": float(predictions.loc[valid_odds, "plcOdds"].max()),
                }
                roi_input = predictions.loc[valid_odds].reset_index(drop=True)
                cumulative, bands = percentile_roi_tables(roi_input)
                cumulative.insert(0, "feature_set", feature_set)
                bands.insert(0, "feature_set", feature_set)
                cumulative.to_csv(
                    TABLE_DIR / f"test_roi_cumulative_{feature_set}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                bands.to_csv(
                    TABLE_DIR / f"test_roi_bands_{feature_set}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                threshold = selection["score_threshold_valid_top10"]
                operational = roi_input.loc[roi_input["score"].ge(threshold)].copy()
                operational = operational.loc[
                    operational.groupby("race_id")["score"].idxmax()
                ].reset_index(drop=True)
                operational["realized_return"] = operational["target"] * operational["plcOdds"] - 1.0
                operational["expected_return"] = operational["calibrated_probability"] * operational["plcOdds"] - 1.0
                ci_low, ci_high = cluster_bootstrap_roi(operational)
                without_top1 = operational.drop(
                    index=operational["realized_return"].idxmax()
                ) if len(operational) else operational
                operational_records.append(
                    {
                        "feature_set": feature_set,
                        "valid_top10_score_threshold": threshold,
                        "bets": int(len(operational)),
                        "races": int(operational["race_id"].nunique()),
                        "hits": int(operational["target"].sum()),
                        "hit_rate": float(operational["target"].mean()) if len(operational) else float("nan"),
                        "mean_plc_odds": float(operational["plcOdds"].mean()) if len(operational) else float("nan"),
                        "expected_roi": float(operational["expected_return"].mean()) if len(operational) else float("nan"),
                        "realized_roi": float(operational["realized_return"].mean()) if len(operational) else float("nan"),
                        "roi_ci_low": ci_low,
                        "roi_ci_high": ci_high,
                        "roi_without_top1": float(without_top1["realized_return"].mean()) if len(without_top1) else float("nan"),
                    }
                )
                operational.to_csv(
                    PREDICTION_DIR / f"test_operational_top1_{feature_set}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    pd.DataFrame(metric_records).to_csv(
        TABLE_DIR / "test_model_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(operational_records).to_csv(
        TABLE_DIR / "test_operational_summary.csv", index=False, encoding="utf-8-sig"
    )
    (TABLE_DIR / "test_data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker = {
        "locked_config_sha256": locked["sha256"],
        "test_evaluation_completed": True,
        "retuning_allowed": False,
    }
    (CONFIG_DIR / "test_evaluation_marker.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] single locked test evaluation complete. config={locked['sha256']}")


if __name__ == "__main__":
    main()
