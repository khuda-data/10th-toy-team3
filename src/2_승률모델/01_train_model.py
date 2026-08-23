"""
01_train_model.py — 모델 학습 파이프라인 (1~3단계)

1단계: 데이터 로드 및 Train/Valid/Test 분할
2단계: 로지스틱회귀 / 랜덤포레스트 / XGBoost 학습 및 평가
3단계: Threshold 튜닝 (valid set에서 탐색 → test set에서 최종 평가)

실행:
    python src/1_전처리/01_train_model.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 프로젝트 경로를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    EXCLUDE_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    RANDOM_STATE,
    get_feature_cols,
    setup_logging,
    ensure_dirs,
)

logger = setup_logging()


# ============================================================
# 1단계: 데이터 로드 및 분할
# ============================================================

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """model_features.csv를 로드하고 fold 기준으로 분할한다."""
    logger.info("=" * 60)
    logger.info("[1단계] 데이터 로드 및 분할")
    logger.info("=" * 60)

    filepath = DATA_DIR / "model_features.csv"
    df = pd.read_csv(filepath, dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str})
    logger.info(f"로드 완료: {filepath.name} ({len(df):,}행 × {len(df.columns)}컬럼)")

    # fold 기준 분할
    train = df[df["fold"] == "train"].copy()
    valid = df[df["fold"] == "valid"].copy()
    test = df[df["fold"] == "test"].copy()

    # 분할 정보 로그
    for name, subset in [("train", train), ("valid", valid), ("test", test)]:
        win_rate = subset[TARGET_COL].mean() * 100
        date_min = subset["rcDate"].min()
        date_max = subset["rcDate"].max()
        logger.info(
            f"  {name:5s}: {len(subset):>6,}행 | "
            f"기간 {date_min}~{date_max} | "
            f"win 비율 {win_rate:.2f}%"
        )

    return train, valid, test


# ============================================================
# 데이터 전처리
# ============================================================

def preprocess(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """피처 전처리 — 결측치 처리 + 범주형 인코딩.

    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test, encoders_dict
    """
    logger.info("피처 전처리 시작")

    # 타겟 분리
    y_train = train[TARGET_COL].values
    y_valid = valid[TARGET_COL].values
    y_test = test[TARGET_COL].values

    # 피처만 추출
    X_train_df = train[feature_cols].copy()
    X_valid_df = valid[feature_cols].copy()
    X_test_df = test[feature_cols].copy()

    # 범주형/수치형 분리
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS]
    num_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    logger.info(f"  수치형 피처: {len(num_cols)}개")
    logger.info(f"  범주형 피처: {len(cat_cols)}개")

    # --- 수치형 결측치: train 중앙값으로 채움 ---
    medians = X_train_df[num_cols].median()
    X_train_df[num_cols] = X_train_df[num_cols].fillna(medians)
    X_valid_df[num_cols] = X_valid_df[num_cols].fillna(medians)
    X_test_df[num_cols] = X_test_df[num_cols].fillna(medians)

    # --- 범주형 처리: 결측 → 'MISSING', LabelEncoder ---
    label_encoders = {}
    for col in cat_cols:
        X_train_df[col] = X_train_df[col].fillna("MISSING").astype(str)
        X_valid_df[col] = X_valid_df[col].fillna("MISSING").astype(str)
        X_test_df[col] = X_test_df[col].fillna("MISSING").astype(str)

        le = LabelEncoder()
        # train에 있는 모든 고유값 + valid/test에만 있는 미지 값 처리
        all_values = sorted(
            set(X_train_df[col].unique())
            | set(X_valid_df[col].unique())
            | set(X_test_df[col].unique())
        )
        le.fit(all_values)
        X_train_df[col] = le.transform(X_train_df[col])
        X_valid_df[col] = le.transform(X_valid_df[col])
        X_test_df[col] = le.transform(X_test_df[col])
        label_encoders[col] = le

    logger.info("  전처리 완료")

    # encoders 저장 (02 스크립트에서 재사용)
    encoders_dict = {
        "label_encoders": label_encoders,
        "medians": medians,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }

    return (
        X_train_df.values.astype(np.float32),
        X_valid_df.values.astype(np.float32),
        X_test_df.values.astype(np.float32),
        y_train,
        y_valid,
        y_test,
        encoders_dict,
    )


# ============================================================
# 2단계: 모델 학습 및 평가
# ============================================================

def evaluate_model(model, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    """모델을 평가하여 지표 딕셔너리를 반환한다."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1_Macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y, y_proba),
    }


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """세 모델을 학습하고 test set 성능을 비교한다."""
    logger.info("=" * 60)
    logger.info("[2단계] 모델 학습 및 평가")
    logger.info("=" * 60)

    # 클래스 불균형 비율
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count
    logger.info(f"  클래스 비율 — 0:{neg_count:,} / 1:{pos_count:,} (비율 {scale_pos:.2f}:1)")

    models = {
        "Logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            verbosity=0,
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        logger.info(f"  학습 중: {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        logger.info(
            f"    → Acc={metrics['Accuracy']:.4f} | "
            f"Prec={metrics['Precision']:.4f} | "
            f"Rec={metrics['Recall']:.4f} | "
            f"F1m={metrics['F1_Macro']:.4f} | "
            f"AUC={metrics['ROC_AUC']:.4f}"
        )

    # 비교표
    df_results = pd.DataFrame(results).T
    df_results.index.name = "Model"
    logger.info("\n" + df_results.to_string())

    return trained_models, df_results


# ============================================================
# 3단계: Threshold 튜닝
# ============================================================

def tune_threshold(
    model,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> tuple[float, pd.DataFrame]:
    """valid set에서 최적 threshold를 탐색하고 test set에서 최종 평가한다."""
    logger.info("=" * 60)
    logger.info(f"[3단계] Threshold 튜닝 (모델: {model_name})")
    logger.info("=" * 60)

    y_proba_valid = model.predict_proba(X_valid)[:, 1]

    thresholds = np.arange(0.05, 0.55, 0.05)
    tuning_results = []

    for thr in thresholds:
        y_pred = (y_proba_valid >= thr).astype(int)
        f1m = f1_score(y_valid, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_valid, y_pred, zero_division=0)
        rec = recall_score(y_valid, y_pred, zero_division=0)
        tuning_results.append({
            "threshold": round(thr, 2),
            "Precision": prec,
            "Recall": rec,
            "F1_Macro": f1m,
        })

    df_tuning = pd.DataFrame(tuning_results)
    best_idx = df_tuning["F1_Macro"].idxmax()
    best_thr = df_tuning.loc[best_idx, "threshold"]

    logger.info(f"  최적 threshold (valid): {best_thr} (F1_Macro={df_tuning.loc[best_idx, 'F1_Macro']:.4f})")

    # test set 비교: 기본(0.5) vs 최적
    metrics_default = evaluate_model(model, X_test, y_test, threshold=0.5)
    metrics_tuned = evaluate_model(model, X_test, y_test, threshold=best_thr)

    comparison = pd.DataFrame({
        "threshold=0.5": metrics_default,
        f"threshold={best_thr}": metrics_tuned,
    }).T

    logger.info("\n  [Test Set 비교]")
    logger.info("\n" + comparison.to_string())

    return best_thr, df_tuning


# ============================================================
# 메인 실행
# ============================================================

def main():
    ensure_dirs()

    # --- 1단계 ---
    train, valid, test = load_and_split()

    # 피처 컬럼 결정
    feature_cols = get_feature_cols(train)
    logger.info(f"\n사용 피처: {len(feature_cols)}개")
    logger.info(f"  {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    # 전처리
    X_train, X_valid, X_test, y_train, y_valid, y_test, encoders_dict = preprocess(
        train, valid, test, feature_cols
    )

    # --- 2단계 ---
    trained_models, df_results = train_models(X_train, y_train, X_test, y_test)

    # 성능 비교표 저장
    df_results.to_csv(RESULTS_DIR / "model_comparison.csv")
    logger.info(f"  저장: results/model_comparison.csv")

    # best 모델 선택 (ROC-AUC 기준)
    best_model_name = df_results["ROC_AUC"].idxmax()
    best_model = trained_models[best_model_name]
    logger.info(f"\n  Best 모델: {best_model_name} (ROC-AUC={df_results.loc[best_model_name, 'ROC_AUC']:.4f})")

    # --- 3단계 ---
    best_thr, df_tuning = tune_threshold(
        best_model, X_valid, y_valid, X_test, y_test, best_model_name
    )

    # 결과 저장
    df_tuning.to_csv(RESULTS_DIR / "threshold_tuning.csv", index=False)
    logger.info(f"  저장: results/threshold_tuning.csv")

    # 모델 저장
    for name, model in trained_models.items():
        model_path = MODELS_DIR / f"{name.lower()}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  모델 저장: {model_path.name}")

    # best 모델 별도 저장 (02 스크립트에서 사용)
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # best threshold 저장
    with open(MODELS_DIR / "best_threshold.pkl", "wb") as f:
        pickle.dump({"model_name": best_model_name, "threshold": best_thr}, f)

    # encoders 저장
    with open(MODELS_DIR / "encoders.pkl", "wb") as f:
        pickle.dump(encoders_dict, f)

    logger.info("\n" + "=" * 60)
    logger.info("01_train_model.py 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
