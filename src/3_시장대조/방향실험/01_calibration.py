"""
방향 1: Calibration 보정

기존 RandomForest 모델의 predict_proba를 Platt Scaling(시그모이드)과
Isotonic Regression으로 보정하여, 예측확률이 실제 승률에 가까워지는지 검증한다.

실행:
    python src/1_전처리/01_calibration.py

출력:
    results/v1_calibration/
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR,
    EXCLUDE_COLS, CATEGORICAL_COLS, TARGET_COL, RANDOM_STATE,
    get_feature_cols, setup_logging, setup_plot_style,
)

logger = setup_logging()

OUTPUT_DIR = RESULTS_DIR / "v1_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_and_model():
    """데이터 및 모델/인코더 로드."""
    df = pd.read_csv(
        DATA_DIR / "model_features.csv",
        dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str},
    )
    with open(MODELS_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return df, model, encoders


def preprocess_split(df, encoders, fold_name):
    """특정 fold의 X, y를 전처리하여 반환."""
    subset = df[df["fold"] == fold_name].copy()
    feature_cols = encoders["feature_cols"]
    cat_cols = encoders["cat_cols"]
    num_cols = encoders["num_cols"]
    medians = encoders["medians"]
    label_encoders = encoders["label_encoders"]

    X_df = subset[feature_cols].copy()
    X_df[num_cols] = X_df[num_cols].fillna(medians)

    for col in cat_cols:
        X_df[col] = X_df[col].fillna("MISSING").astype(str)
        le = label_encoders[col]
        known = set(le.classes_)
        X_df[col] = X_df[col].apply(lambda x: x if x in known else "MISSING")
        X_df[col] = le.transform(X_df[col])

    y = subset[TARGET_COL].values
    return X_df.values.astype(np.float32), y


def evaluate(y_true, y_proba, threshold=0.5):
    """지표 계산."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }


def main():
    logger.info("=" * 60)
    logger.info("[방향 1] Calibration 보정")
    logger.info("=" * 60)

    df, base_model, encoders = load_data_and_model()

    X_train, y_train = preprocess_split(df, encoders, "train")
    X_valid, y_valid = preprocess_split(df, encoders, "valid")
    X_test, y_test = preprocess_split(df, encoders, "test")

    # --- 보정 모델 학습 (valid set 사용) ---
    # sklearn >= 1.6에서 cv="prefit" 제거됨. 수동으로 보정 구현.
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as LR_Cal

    # 기본 모델의 valid set 예측확률
    raw_proba_valid = base_model.predict_proba(X_valid)[:, 1]

    # Platt Scaling: 로지스틱회귀로 확률→확률 매핑
    logger.info("\n  Platt Scaling (sigmoid) 보정 중...")
    platt_model = LR_Cal(max_iter=1000)
    platt_model.fit(raw_proba_valid.reshape(-1, 1), y_valid)

    # Isotonic Regression
    logger.info("  Isotonic Regression 보정 중...")
    iso_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_model.fit(raw_proba_valid, y_valid)

    # 보정 적용 함수
    class PlattCalibrator:
        def __init__(self, base, platt):
            self.base = base
            self.platt = platt
        def predict_proba(self, X):
            raw = self.base.predict_proba(X)[:, 1]
            cal = self.platt.predict_proba(raw.reshape(-1, 1))
            return cal

    class IsotonicCalibrator:
        def __init__(self, base, iso):
            self.base = base
            self.iso = iso
        def predict_proba(self, X):
            raw = self.base.predict_proba(X)[:, 1]
            cal = self.iso.predict(raw)
            return np.column_stack([1 - cal, cal])

    cal_sigmoid = PlattCalibrator(base_model, platt_model)
    cal_isotonic = IsotonicCalibrator(base_model, iso_model)

    # --- Test set 평가 ---
    probas = {
        "Raw (before cal)": base_model.predict_proba(X_test)[:, 1],
        "Platt Scaling": cal_sigmoid.predict_proba(X_test)[:, 1],
        "Isotonic": cal_isotonic.predict_proba(X_test)[:, 1],
    }

    # 시장 확률 가져오기
    df_odds = pd.read_csv(DATA_DIR / "market_odds.csv")
    test_entries = df[df["fold"] == "test"][["entry_id", "race_id"]].copy()
    test_entries = test_entries.merge(df_odds[["entry_id", "winOdds"]], on="entry_id", how="left")
    test_entries["inv_odds"] = 1.0 / test_entries["winOdds"]
    test_entries["market_prob"] = test_entries.groupby("race_id")["inv_odds"].transform(
        lambda x: x / x.sum()
    )
    market_prob = test_entries["market_prob"].values

    # --- 성능 비교표 ---
    results = {}
    for name, proba in probas.items():
        results[name] = evaluate(y_test, proba)
        # 실제 승률과의 평균 절대 오차 (calibration 품질)
        # 10분위 기준
        fraction_pos, mean_predicted = calibration_curve(y_test, proba, n_bins=10)
        results[name]["Cal_MAE"] = np.mean(np.abs(fraction_pos - mean_predicted))

    # 시장 확률도 비교
    results["Market Odds"] = evaluate(y_test, market_prob)
    fraction_pos, mean_predicted = calibration_curve(y_test, market_prob, n_bins=10)
    results["Market Odds"]["Cal_MAE"] = np.mean(np.abs(fraction_pos - mean_predicted))

    df_results = pd.DataFrame(results).T
    df_results.index.name = "Model"
    logger.info(f"\n{df_results.round(4).to_string()}")

    df_results.to_csv(OUTPUT_DIR / "calibration_comparison.csv")
    logger.info(f"\n  저장: results/v1_calibration/calibration_comparison.csv")

    # --- Calibration Curve 시각화 ---
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, proba in probas.items():
        fraction_pos, mean_predicted = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_predicted, fraction_pos, "o-", label=name)

    # 시장 확률
    fraction_pos, mean_predicted = calibration_curve(y_test, market_prob, n_bins=10)
    ax.plot(mean_predicted, fraction_pos, "s-", label="Market Odds", linewidth=2)

    ax.set_xlabel("Predicted Probability (mean)")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title("Calibration Curve — Before/After vs Market")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_curve.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  저장: results/v1_calibration/calibration_curve.png")

    # --- 확률 분포 비교 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    all_probas = list(probas.items()) + [("Market Odds", market_prob)]

    for ax, (name, proba) in zip(axes.flatten(), all_probas):
        ax.hist(proba, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{name}\nmean={proba.mean():.4f}, std={proba.std():.4f}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.axvline(y_test.mean(), color="red", linestyle="--", label=f"Actual win rate={y_test.mean():.4f}")
        ax.legend(fontsize=8)

    plt.suptitle("Probability Distribution Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "probability_distributions.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  저장: results/v1_calibration/probability_distributions.png")

    # --- 보정 모델 저장 (원본 보정기만 저장) ---
    with open(OUTPUT_DIR / "cal_sigmoid.pkl", "wb") as f:
        pickle.dump(platt_model, f)
    with open(OUTPUT_DIR / "cal_isotonic.pkl", "wb") as f:
        pickle.dump(iso_model, f)

    logger.info("\n" + "=" * 60)
    logger.info("[방향 1] 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
