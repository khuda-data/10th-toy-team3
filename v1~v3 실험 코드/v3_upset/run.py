"""
방향 3: 이변 예측 모델 (upset_B 타겟 변경)

"승률을 정확히 맞추겠다"가 아니라 "시장이 틀리는 경우를 찾겠다"로 목표를 변경.
타겟: upset_B = 인기 하위 50%(pop_pct >= 0.5)이면서 1착한 말

이 모델은 "비인기마인데 이길 가능성이 높은 말"을 탐지하는 것이 목적.

실행:
    python src/pipeline/v3_upset/run.py

출력:
    results/v3_upset/
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR,
    EXCLUDE_COLS, CATEGORICAL_COLS, RANDOM_STATE,
    get_feature_cols, setup_logging, setup_plot_style,
)

logger = setup_logging()

OUTPUT_DIR = RESULTS_DIR / "v3_upset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# upset_B 타겟: 인기 하위 50%이면서 1착
TARGET_COL = "upset_B"


def load_data():
    """데이터 로드. upset_B는 race_outcome에 있으므로 조인 필요."""
    df = pd.read_csv(
        DATA_DIR / "model_features.csv",
        dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str},
    )
    df_outcome = pd.read_csv(DATA_DIR / "race_outcome.csv")
    df_odds = pd.read_csv(DATA_DIR / "market_odds.csv")

    # upset_B 조인
    df = df.merge(df_outcome[["entry_id", "upset_B"]], on="entry_id", how="left")
    # pop_pct 조인 (필터링용)
    df = df.merge(df_odds[["entry_id", "pop_pct", "winOdds"]], on="entry_id", how="left")

    return df


def preprocess(df, fold_name, encoders):
    """전처리 + 비인기마(pop_pct >= 0.5)만 필터링."""
    subset = df[df["fold"] == fold_name].copy()

    # 비인기마만 필터 (upset_B는 pop_pct >= 0.5인 말에서만 의미있음)
    subset = subset[subset["pop_pct"] >= 0.5].reset_index(drop=True)

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
    return X_df.values.astype(np.float32), y, subset


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
    logger.info("=" * 60)
    logger.info("[방향 3] 이변 예측 모델 (upset_B)")
    logger.info("=" * 60)

    df = load_data()

    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    # 분할 (비인기마만)
    X_train, y_train, train_df = preprocess(df, "train", encoders)
    X_valid, y_valid, valid_df = preprocess(df, "valid", encoders)
    X_test, y_test, test_df = preprocess(df, "test", encoders)

    logger.info(f"  비인기마(pop_pct>=0.5) 필터 후:")
    logger.info(f"    train: {len(y_train):,}행 (upset_B 비율: {y_train.mean():.4f})")
    logger.info(f"    valid: {len(y_valid):,}행 (upset_B 비율: {y_valid.mean():.4f})")
    logger.info(f"    test:  {len(y_test):,}행 (upset_B 비율: {y_test.mean():.4f})")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    logger.info(f"    클래스 비율: 0={neg:,} / 1={pos:,} (비율 {neg/max(pos,1):.1f}:1)")

    # --- 모델 학습 ---
    models = {
        "RF_upset": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGB_upset": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=neg / max(pos, 1), eval_metric="auc",
            random_state=RANDOM_STATE, verbosity=0
        ),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        logger.info(f"\n  학습 중: {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)
        results[name] = metrics
        logger.info(
            f"    AUC={metrics['ROC_AUC']:.4f} | "
            f"F1m={metrics['F1_Macro']:.4f} | "
            f"Prec={metrics['Precision']:.4f} | "
            f"Rec={metrics['Recall']:.4f}"
        )

    # best 선택
    df_results = pd.DataFrame(results).T
    df_results.index.name = "Model"
    best_name = df_results["ROC_AUC"].idxmax()
    best_model = trained[best_name]

    logger.info(f"\n  Best: {best_name} (AUC={df_results.loc[best_name, 'ROC_AUC']:.4f})")

    # --- Threshold 탐색 (valid) ---
    proba_valid = best_model.predict_proba(X_valid)[:, 1]
    best_thr = 0.5
    best_f1 = 0

    for thr in np.arange(0.05, 0.55, 0.05):
        f1 = f1_score(y_valid, (proba_valid >= thr).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = round(thr, 2)

    logger.info(f"  최적 threshold (valid): {best_thr} (F1m={best_f1:.4f})")

    # --- Test 최종 평가 ---
    proba_test = best_model.predict_proba(X_test)[:, 1]
    final_metrics = evaluate(y_test, proba_test, threshold=best_thr)
    results[f"{best_name}_tuned(thr={best_thr})"] = final_metrics

    df_results = pd.DataFrame(results).T
    logger.info(f"\n{df_results.round(4).to_string()}")
    df_results.to_csv(OUTPUT_DIR / "upset_model_comparison.csv")
    logger.info(f"\n  저장: results/v3_upset/upset_model_comparison.csv")

    # --- 실용 가치 분석: 모델이 찾은 이변 후보의 실제 적중률 ---
    test_df = test_df.copy()
    test_df["upset_proba"] = proba_test
    test_df["upset_pred"] = (proba_test >= best_thr).astype(int)

    # 모델이 "이변 가능성 높음"으로 예측한 말들의 실제 성적
    flagged = test_df[test_df["upset_pred"] == 1]
    not_flagged = test_df[test_df["upset_pred"] == 0]

    logger.info(f"\n  --- 실용 가치 분석 ---")
    logger.info(f"  모델 플래그 ON: {len(flagged):,}건")
    logger.info(f"    실제 upset_B 비율: {flagged['upset_B'].mean():.4f} ({flagged['upset_B'].sum():.0f}건)")
    logger.info(f"  모델 플래그 OFF: {len(not_flagged):,}건")
    logger.info(f"    실제 upset_B 비율: {not_flagged['upset_B'].mean():.4f} ({not_flagged['upset_B'].sum():.0f}건)")

    if len(flagged) > 0:
        lift = flagged["upset_B"].mean() / max(y_test.mean(), 1e-8)
        logger.info(f"  Lift (플래그 vs 전체): {lift:.2f}배")

        # 평균 배당률
        avg_odds_flagged = flagged["winOdds"].mean()
        avg_odds_all = test_df["winOdds"].mean()
        logger.info(f"  플래그 평균 배당: {avg_odds_flagged:.1f}배 (전체 평균: {avg_odds_all:.1f}배)")

    # --- Feature Importance ---
    setup_plot_style()
    if hasattr(best_model, "feature_importances_"):
        feature_cols = encoders["feature_cols"]
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top20 = fi.head(20)
        ax.barh(range(len(top20)), top20["importance"].values, color="darkorange")
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Upset Model ({best_name}) — Feature Importance Top 20")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "upset_feature_importance.png", bbox_inches="tight")
        plt.close()
        logger.info(f"  저장: results/v3_upset/upset_feature_importance.png")

        fi.to_csv(OUTPUT_DIR / "upset_feature_importance.csv", index=False)

    # --- 이변 확률 상위 구간별 적중률 ---
    test_df["proba_decile"] = pd.qcut(test_df["upset_proba"], q=10, labels=False, duplicates="drop")
    decile_stats = test_df.groupby("proba_decile").agg(
        upset_rate=("upset_B", "mean"),
        avg_odds=("winOdds", "mean"),
        count=("entry_id", "count"),
    ).reset_index()

    decile_stats.to_csv(OUTPUT_DIR / "upset_decile_analysis.csv", index=False)
    logger.info(f"\n  예측확률 10분위별 이변 적중률:")
    logger.info(f"\n{decile_stats.round(4).to_string(index=False)}")

    # 시각화
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = decile_stats["proba_decile"]
    ax1.bar(x, decile_stats["upset_rate"], color="darkorange", alpha=0.7, label="Upset Hit Rate")
    ax1.set_xlabel("Prediction Score Decile (0=low -> 9=high)")
    ax1.set_ylabel("Actual upset_B Rate", color="darkorange")
    ax1.axhline(y_test.mean(), color="red", linestyle="--", label=f"Overall avg ({y_test.mean():.4f})")

    ax2 = ax1.twinx()
    ax2.plot(x, decile_stats["avg_odds"], "b-o", label="Avg winOdds")
    ax2.set_ylabel("Avg winOdds", color="blue")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Upset Model — Hit Rate & Odds by Decile")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "upset_decile_chart.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  저장: results/v3_upset/upset_decile_chart.png")

    # 모델 저장
    with open(OUTPUT_DIR / "best_upset_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    logger.info("\n" + "=" * 60)
    logger.info("[방향 3] 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
