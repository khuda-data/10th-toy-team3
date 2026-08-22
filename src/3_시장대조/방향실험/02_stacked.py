"""
방향 2: 2단계 모델 (시장확률 + 모델확률 결합)

1단계 모델의 예측확률과 시장 확률(q)을 결합한 2단계 모델을 만들어
시장보다 나은 확률 추정이 가능한지 검증한다.

실행:
    python src/1_전처리/02_stacked.py

출력:
    results/v2_stacked/
"""

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
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR,
    TARGET_COL, RANDOM_STATE,
    setup_logging, setup_plot_style,
)

logger = setup_logging()

OUTPUT_DIR = RESULTS_DIR / "v2_stacked"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """데이터 로드 + 시장 확률 조인."""
    df = pd.read_csv(
        DATA_DIR / "model_features.csv",
        dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str},
    )
    df_odds = pd.read_csv(DATA_DIR / "market_odds.csv")

    # 시장 확률 계산
    odds_sub = df_odds[["entry_id", "winOdds", "q", "pop_pct", "is_fav"]].copy()
    df = df.merge(odds_sub, on="entry_id", how="left")

    # 경주 내 정규화 시장 확률
    df["inv_odds"] = 1.0 / df["winOdds"]
    df["market_prob"] = df.groupby("race_id")["inv_odds"].transform(lambda x: x / x.sum())

    return df


def prepare_stage1_proba(df, encoders, base_model):
    """1단계 모델의 예측확률을 계산."""
    feature_cols = encoders["feature_cols"]
    cat_cols = encoders["cat_cols"]
    num_cols = encoders["num_cols"]
    medians = encoders["medians"]
    label_encoders = encoders["label_encoders"]

    X_df = df[feature_cols].copy()
    X_df[num_cols] = X_df[num_cols].fillna(medians)

    for col in cat_cols:
        X_df[col] = X_df[col].fillna("MISSING").astype(str)
        le = label_encoders[col]
        known = set(le.classes_)
        X_df[col] = X_df[col].apply(lambda x: x if x in known else "MISSING")
        X_df[col] = le.transform(X_df[col])

    proba = base_model.predict_proba(X_df.values.astype(np.float32))[:, 1]
    return proba


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
    logger.info("[방향 2] 2단계 모델 (시장확률 + 모델확률 결합)")
    logger.info("=" * 60)

    # 데이터 로드
    df = load_data()

    with open(MODELS_DIR / "best_model.pkl", "rb") as f:
        base_model = pickle.load(f)
    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    # 1단계 예측확률 추가
    logger.info("  1단계 모델 예측확률 계산 중...")
    df["model_prob"] = prepare_stage1_proba(df, encoders, base_model)

    # 2단계 피처: 모델확률, 시장확률, 괴리, 인기순위 등
    stage2_features = ["model_prob", "market_prob", "q", "pop_pct", "is_fav"]

    # 괴리 피처 추가
    df["gap"] = df["model_prob"] - df["market_prob"]
    df["gap_abs"] = df["gap"].abs()
    df["prob_ratio"] = df["model_prob"] / (df["market_prob"] + 1e-8)
    stage2_features += ["gap", "gap_abs", "prob_ratio"]

    logger.info(f"  2단계 피처: {stage2_features}")

    # 분할
    train = df[df["fold"] == "train"]
    valid = df[df["fold"] == "valid"]
    test = df[df["fold"] == "test"]

    X_train = train[stage2_features].values.astype(np.float32)
    X_valid = valid[stage2_features].values.astype(np.float32)
    X_test = test[stage2_features].values.astype(np.float32)
    y_train = train[TARGET_COL].values
    y_valid = valid[TARGET_COL].values
    y_test = test[TARGET_COL].values

    # NaN 처리
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_valid = np.nan_to_num(X_valid, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    # --- 2단계 모델 학습 ---
    models = {
        "Stage2_Logistic": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Stage2_RF": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=30,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Stage2_XGB": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=neg / pos, eval_metric="auc",
            random_state=RANDOM_STATE, verbosity=0
        ),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        logger.info(f"  학습 중: {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)
        fraction_pos, mean_predicted = calibration_curve(y_test, proba, n_bins=10)
        metrics["Cal_MAE"] = np.mean(np.abs(fraction_pos - mean_predicted))
        results[name] = metrics

        logger.info(f"    AUC={metrics['ROC_AUC']:.4f} | F1m={metrics['F1_Macro']:.4f} | Cal_MAE={metrics['Cal_MAE']:.4f}")

    # test에서 1단계 vs 시장 vs 2단계 비교
    market_test = test["market_prob"].values
    model1_test = test["model_prob"].values

    results["1stage_RF"] = evaluate(y_test, model1_test)
    fraction_pos, mean_predicted = calibration_curve(y_test, model1_test, n_bins=10)
    results["1stage_RF"]["Cal_MAE"] = np.mean(np.abs(fraction_pos - mean_predicted))

    results["Market Odds"] = evaluate(y_test, market_test)
    fraction_pos, mean_predicted = calibration_curve(y_test, market_test, n_bins=10)
    results["Market Odds"]["Cal_MAE"] = np.mean(np.abs(fraction_pos - mean_predicted))

    # 결과표
    df_results = pd.DataFrame(results).T
    df_results.index.name = "Model"
    logger.info(f"\n{df_results.round(4).to_string()}")

    df_results.to_csv(OUTPUT_DIR / "stacked_comparison.csv")
    logger.info(f"\n  저장: results/v2_stacked/stacked_comparison.csv")

    # --- Calibration Curve ---
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # 최고 2단계 모델
    best_name = df_results.loc[
        [n for n in df_results.index if n.startswith("Stage2")], "ROC_AUC"
    ].idxmax()
    best_stage2 = trained[best_name]
    best_proba = best_stage2.predict_proba(X_test)[:, 1]

    for label, proba in [
        ("Stage1 RF (baseline)", model1_test),
        ("Market Odds", market_test),
        (f"Stage2 {best_name}", best_proba),
    ]:
        fraction_pos, mean_predicted = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_predicted, fraction_pos, "o-", label=label)

    ax.set_xlabel("Predicted Probability (mean)")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title("Calibration Curve — Stage1 vs Stage2 vs Market")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_curve.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  저장: results/v2_stacked/calibration_curve.png")

    # --- Feature Importance (2단계 최고 모델) ---
    if hasattr(best_stage2, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": stage2_features,
            "importance": best_stage2.feature_importances_,
        }).sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(fi)), fi["importance"].values, color="steelblue")
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Stage2 Model ({best_name}) Feature Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "stage2_feature_importance.png", bbox_inches="tight")
        plt.close()
        logger.info(f"  저장: results/v2_stacked/stage2_feature_importance.png")

    # 모델 저장
    with open(OUTPUT_DIR / "best_stage2.pkl", "wb") as f:
        pickle.dump(best_stage2, f)

    logger.info("\n" + "=" * 60)
    logger.info("[방향 2] 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
