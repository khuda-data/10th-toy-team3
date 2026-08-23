"""
03_train_compare.py — A/B/C 세 피처셋 × 2 모델 비교

A = q(배당률) 단독
B = 기존 피처만 (배당률 제외)
C = q + 기존 피처 (결합)

각각 Logistic Regression, Random Forest로 학습하여 test set에서 비교.

실행:
    python src/4_이변모델/03_train_compare.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))  # config.py 는 같은 폴더
from config import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds")


def evaluate(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[Step 3] Train & Compare A / B / C")
    logger.info("=" * 60)

    # Load prepared data
    with open(OUTPUT_DIR / "prepared_data.pkl", "rb") as f:
        prep = pickle.load(f)

    df = prep["df"]
    a_features = prep["a_features"]
    b_features = prep["b_features"]
    c_features = prep["c_features"]

    # Split
    train = df[df["fold"] == "train"]
    test = df[df["fold"] == "test"]

    y_train = train["upset"].values
    y_test = test["upset"].values

    logger.info(f"  Train: {len(train):,} | Test: {len(test):,}")
    logger.info(f"  Train upset rate: {y_train.mean():.4f}")
    logger.info(f"  Test upset rate:  {y_test.mean():.4f}")

    # Models
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    model_configs = {
        "Logistic": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    # Feature sets
    feature_sets = {
        "A (q only)": a_features,
        "B (no odds)": b_features,
        "C (q + features)": c_features,
    }

    results = []
    trained_models = {}

    for fs_name, features in feature_sets.items():
        X_train = train[features].values.astype(np.float32)
        X_test = test[features].values.astype(np.float32)

        for model_name, model_template in model_configs.items():
            # Clone model (fresh instance)
            import sklearn.base
            model = sklearn.base.clone(model_template)

            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, proba)

            key = f"{fs_name}_{model_name}"
            trained_models[key] = model

            row = {
                "feature_set": fs_name,
                "model": model_name,
                "n_features": len(features),
                **metrics,
            }
            results.append(row)

            logger.info(
                f"  {fs_name:20s} + {model_name:10s} | "
                f"AUC={metrics['ROC_AUC']:.4f} | F1m={metrics['F1_Macro']:.4f} | "
                f"Prec={metrics['Precision']:.4f} | Rec={metrics['Recall']:.4f}"
            )

    # Results table
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    logger.info(f"\n  Saved: results/upset_with_odds/model_comparison.csv")

    # Print comparison
    logger.info(f"\n{'='*80}")
    logger.info(df_results.to_string(index=False))
    logger.info(f"{'='*80}")

    # Best C model
    c_results = df_results[df_results["feature_set"] == "C (q + features)"]
    best_c = c_results.loc[c_results["ROC_AUC"].idxmax()]
    logger.info(f"\n  Best C model: {best_c['model']} (AUC={best_c['ROC_AUC']:.4f})")

    # Save trained models
    with open(OUTPUT_DIR / "trained_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)

    # Key comparison: A vs C improvement
    a_best = df_results[df_results["feature_set"] == "A (q only)"]["ROC_AUC"].max()
    b_best = df_results[df_results["feature_set"] == "B (no odds)"]["ROC_AUC"].max()
    c_best = best_c["ROC_AUC"]

    logger.info(f"\n  --- Key Comparison ---")
    logger.info(f"  A (q only) best AUC:      {a_best:.4f}")
    logger.info(f"  B (no odds) best AUC:     {b_best:.4f}")
    logger.info(f"  C (q + features) best AUC: {c_best:.4f}")
    logger.info(f"  C vs A improvement:        {c_best - a_best:+.4f}")
    logger.info(f"  C vs B improvement:        {c_best - b_best:+.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("03_train_compare.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
