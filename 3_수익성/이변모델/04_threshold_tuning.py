"""
04_threshold_tuning.py — Best C 모델 Threshold 튜닝

valid set에서 F1(Macro) 최대가 되는 임계값을 탐색하고,
test set에서 튜닝 전후를 비교한다.

실행:
    python src/4_이변모델/04_threshold_tuning.py
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds")


def evaluate(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }


def main():
    logger.info("=" * 60)
    logger.info("[Step 4] Threshold Tuning (C model)")
    logger.info("=" * 60)

    # Load
    with open(OUTPUT_DIR / "prepared_data.pkl", "rb") as f:
        prep = pickle.load(f)
    with open(OUTPUT_DIR / "trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)

    df = prep["df"]
    c_features = prep["c_features"]

    # Determine best C model from comparison
    comp = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    c_rows = comp[comp["feature_set"] == "C (q + features)"]
    best_model_name = c_rows.loc[c_rows["ROC_AUC"].idxmax(), "model"]
    model_key = f"C (q + features)_{best_model_name}"
    model = trained_models[model_key]

    logger.info(f"  Best C model: {best_model_name}")

    # Valid set for threshold search
    valid = df[df["fold"] == "valid"]
    test = df[df["fold"] == "test"]

    X_valid = valid[c_features].values.astype(np.float32)
    X_test = test[c_features].values.astype(np.float32)
    y_valid = valid["upset"].values
    y_test = test["upset"].values

    proba_valid = model.predict_proba(X_valid)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    # Search threshold on valid
    logger.info(f"\n  Threshold search on valid set ({len(valid):,} rows):")
    tuning_results = []
    for thr in np.arange(0.05, 0.55, 0.05):
        thr = round(thr, 2)
        y_pred = (proba_valid >= thr).astype(int)
        f1m = f1_score(y_valid, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_valid, y_pred, zero_division=0)
        rec = recall_score(y_valid, y_pred, zero_division=0)
        tuning_results.append({"threshold": thr, "F1_Macro": f1m, "Precision": prec, "Recall": rec})
        logger.info(f"    thr={thr:.2f} | F1m={f1m:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")

    df_tuning = pd.DataFrame(tuning_results)
    best_thr = df_tuning.loc[df_tuning["F1_Macro"].idxmax(), "threshold"]
    logger.info(f"\n  Best threshold (valid): {best_thr}")

    # Compare on test: default(0.5) vs tuned
    metrics_default = evaluate(y_test, proba_test, threshold=0.5)
    metrics_tuned = evaluate(y_test, proba_test, threshold=best_thr)

    comparison = pd.DataFrame([metrics_default, metrics_tuned], index=["Default (0.5)", f"Tuned ({best_thr})"])
    logger.info(f"\n  --- Test Set Comparison ---")
    logger.info(f"\n{comparison.to_string()}")

    # Save
    df_tuning.to_csv(OUTPUT_DIR / "threshold_tuning.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "threshold_comparison.csv")
    logger.info(f"\n  Saved: threshold_tuning.csv, threshold_comparison.csv")

    # Save best threshold
    with open(OUTPUT_DIR / "best_threshold.pkl", "wb") as f:
        pickle.dump({"model_name": best_model_name, "threshold": best_thr}, f)

    logger.info("\n" + "=" * 60)
    logger.info("04_threshold_tuning.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
