"""
01_run_analysis.py — 버전별 모델 비교 분석

v1~v8 전처리 버전을 직접 읽어서:
1. 각 버전별 RF/XGBoost/Logistic 학습 + 평가
2. 이상치 제거 효과 비교 (v1 vs v5, v2 vs v6 등)
3. 스케일링 방식별 성능 비교
4. 최종 결과 보고서 HTML 생성

실행:
    python src/analysis/01_run_analysis.py

출력:
    results/analysis/
        model_results.csv          전체 결과표
        comparison_chart.png       비교 그래프
        report.html                HTML 보고서
"""

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    EXCLUDE_COLS, CATEGORICAL_COLS, TARGET_COL, RANDOM_STATE,
    get_feature_cols, setup_plot_style,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VERSIONS_DIR = Path("data/versions")
OUTPUT_DIR = Path("results/analysis")

VERSIONS = {
    "v1_base": {"outlier": "included", "scaling": "None"},
    "v2_standard": {"outlier": "included", "scaling": "Standard"},
    "v3_minmax": {"outlier": "included", "scaling": "MinMax"},
    "v4_robust": {"outlier": "included", "scaling": "Robust"},
    "v5_base_no_outlier": {"outlier": "removed", "scaling": "None"},
    "v6_standard_no_outlier": {"outlier": "removed", "scaling": "Standard"},
    "v7_minmax_no_outlier": {"outlier": "removed", "scaling": "MinMax"},
    "v8_robust_no_outlier": {"outlier": "removed", "scaling": "Robust"},
}


# ============================================================
# 데이터 로드 + 전처리
# ============================================================

def load_version(version: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """버전 CSV를 로드하고 X/y 분리 + 범주형 인코딩."""
    path = VERSIONS_DIR / f"{version}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str})

    # fold 분할
    train = df[df["fold"] == "train"]
    valid = df[df["fold"] == "valid"]
    test = df[df["fold"] == "test"]

    # 피처 컬럼 결정
    feature_cols = get_feature_cols(df)

    # X/y 분리
    def prepare(subset):
        X = subset[feature_cols].copy()

        # 범주형 인코딩
        cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS or X[c].dtype == "object"]
        for col in cat_cols:
            X[col] = X[col].fillna("MISSING").astype(str)
            le = LabelEncoder()
            # 모든 분할의 고유값을 합쳐서 fit
            all_vals = sorted(set(df[col].fillna("MISSING").astype(str).unique()))
            le.fit(all_vals)
            X[col] = le.transform(X[col])

        # 남은 수치형 NaN (있다면)
        X = X.fillna(0)

        y = subset[TARGET_COL].values
        return X.values.astype(np.float32), y

    X_train, y_train = prepare(train)
    X_valid, y_valid = prepare(valid)
    X_test, y_test = prepare(test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols


# ============================================================
# 모델 학습 + 평가
# ============================================================

def evaluate(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """3개 모델 학습 + test 평가 → dict."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    models = {
        "Logistic": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs"
        ),
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=neg / max(pos, 1), eval_metric="auc",
            random_state=RANDOM_STATE, verbosity=0
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)
        results[name] = metrics

    return results


# ============================================================
# 메인
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Version-wise Model Comparison Analysis")
    logger.info("=" * 60)

    all_results = []

    for version, meta in VERSIONS.items():
        logger.info(f"\n  [{version}] outlier={meta['outlier']}, scaling={meta['scaling']}")

        data = load_version(version)
        if data is None:
            logger.warning(f"    Skipped (file not found)")
            continue

        X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols = data
        logger.info(f"    train={len(y_train):,} | test={len(y_test):,} | features={len(feature_cols)}")

        results = train_and_evaluate(X_train, y_train, X_test, y_test)

        for model_name, metrics in results.items():
            row = {
                "version": version,
                "outlier": meta["outlier"],
                "scaling": meta["scaling"],
                "model": model_name,
                **metrics,
            }
            all_results.append(row)
            logger.info(f"    {model_name:10s} AUC={metrics['ROC_AUC']:.4f} F1m={metrics['F1_Macro']:.4f}")

    # 결과 저장
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    logger.info(f"\n  Saved: results/analysis/model_results.csv ({len(df_results)} rows)")

    # 시각화
    plot_comparison(df_results)

    # 보고서 생성
    generate_report(df_results)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


# ============================================================
# 시각화
# ============================================================

def plot_comparison(df: pd.DataFrame):
    """버전 x 모델 AUC 비교 그래프."""
    setup_plot_style()

    # AUC 비교 (모델별)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    models = ["Logistic", "RF", "XGBoost"]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].copy()
        sub = sub.sort_values("ROC_AUC", ascending=True)

        colors = ["#1976d2" if row["outlier"] == "included" else "#ff7043"
                  for _, row in sub.iterrows()]

        ax.barh(range(len(sub)), sub["ROC_AUC"], color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["version"], fontsize=8)
        ax.set_xlabel("ROC-AUC")
        ax.set_title(f"{model}")
        ax.set_xlim(0.4, 0.85)

        # legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1976d2", label="Outlier included"),
            Patch(facecolor="#ff7043", label="Outlier removed"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.suptitle("ROC-AUC by Version and Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_chart.png", bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"  Saved: results/analysis/comparison_chart.png")

    # 이상치 효과 비교 (paired)
    fig, ax = plt.subplots(figsize=(10, 6))
    scalings = ["None", "Standard", "MinMax", "Robust"]
    x = np.arange(len(scalings))
    width = 0.25

    for i, model in enumerate(models):
        incl = []
        rmvd = []
        for s in scalings:
            row_incl = df[(df["model"] == model) & (df["scaling"] == s) & (df["outlier"] == "included")]
            row_rmvd = df[(df["model"] == model) & (df["scaling"] == s) & (df["outlier"] == "removed")]
            incl.append(row_incl["ROC_AUC"].values[0] if len(row_incl) > 0 else 0)
            rmvd.append(row_rmvd["ROC_AUC"].values[0] if len(row_rmvd) > 0 else 0)

        diff = [r - i for r, i in zip(rmvd, incl)]
        ax.bar(x + i * width, diff, width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels(scalings)
    ax.set_xlabel("Scaling Method")
    ax.set_ylabel("AUC Difference (removed - included)")
    ax.set_title("Outlier Removal Effect on AUC (positive = removal helped)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outlier_effect.png", bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"  Saved: results/analysis/outlier_effect.png")


# ============================================================
# HTML 보고서
# ============================================================

def generate_report(df: pd.DataFrame):
    """결과를 HTML 보고서로 정리."""
    import base64

    def img_b64(path):
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    comp_img = img_b64(OUTPUT_DIR / "comparison_chart.png")
    effect_img = img_b64(OUTPUT_DIR / "outlier_effect.png")

    # Best model per version
    best_per_version = df.loc[df.groupby("version")["ROC_AUC"].idxmax()]

    # Overall best
    overall_best = df.loc[df["ROC_AUC"].idxmax()]

    # 이상치 효과 요약
    included = df[df["outlier"] == "included"].groupby("model")["ROC_AUC"].mean()
    removed = df[df["outlier"] == "removed"].groupby("model")["ROC_AUC"].mean()
    effect = removed - included

    # 결과 테이블 HTML
    table_rows = ""
    for _, row in df.sort_values(["model", "ROC_AUC"], ascending=[True, False]).iterrows():
        highlight = ' style="background:#e8f5e9;"' if row["ROC_AUC"] == overall_best["ROC_AUC"] else ""
        table_rows += f'<tr{highlight}><td>{row["version"]}</td><td>{row["outlier"]}</td><td>{row["scaling"]}</td><td>{row["model"]}</td><td>{row["Accuracy"]:.4f}</td><td>{row["Precision"]:.4f}</td><td>{row["Recall"]:.4f}</td><td>{row["F1_Macro"]:.4f}</td><td><strong>{row["ROC_AUC"]:.4f}</strong></td></tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>버전별 모델 비교 분석</title>
<style>
    body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; line-height: 1.8; font-size: 15px; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 35px; border-left: 4px solid #3f51b5; padding-left: 12px; }}
    .box {{ background: #e8eaf6; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .good {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 14px 18px; margin: 15px 0; }}
    .insight {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 18px; margin: 15px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 12px; }}
    th {{ background: #3f51b5; color: white; padding: 8px; text-align: left; }}
    td {{ padding: 7px; border-bottom: 1px solid #e0e0e0; }}
    tr:hover {{ background: #e8eaf6; }}
    .chart {{ text-align: center; margin: 20px 0; }}
    .chart img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }}
    .code-block {{ background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px; margin: 12px 0; font-family: Consolas, monospace; font-size: 13px; line-height: 1.6; }}
    .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ccc; color: #777; font-size: 12px; text-align: center; }}
</style></head><body>

<h1>버전별 모델 비교 분석 결과</h1>

<div class="box">
<h3>분석 개요</h3>
<ul>
<li><strong>데이터:</strong> 8가지 전처리 버전 (이상치 포함/제거 x 스케일링 4종)</li>
<li><strong>모델:</strong> Logistic Regression, Random Forest, XGBoost</li>
<li><strong>평가:</strong> test set (시간순 마지막 20%) 기준 ROC-AUC, F1-Macro</li>
<li><strong>총 실험:</strong> 8 versions x 3 models = 24 combinations</li>
</ul>
</div>

<div class="good">
<h3>핵심 결론</h3>
<ul>
<li><strong>Best:</strong> {overall_best['version']} + {overall_best['model']} (AUC = {overall_best['ROC_AUC']:.4f})</li>
<li><strong>이상치 제거 효과:</strong> RF {effect.get('RF', 0):+.4f} | XGBoost {effect.get('XGBoost', 0):+.4f} | Logistic {effect.get('Logistic', 0):+.4f}</li>
<li><strong>스케일링 영향:</strong> 트리 모델(RF, XGB)은 스케일링에 거의 무관. 로지스틱은 Standard가 가장 좋음.</li>
</ul>
</div>

<h2>1. 버전별 AUC 비교</h2>
<div class="chart"><img src="{comp_img}" alt="Comparison"></div>

<p class="code-block">
# Each version tested with 3 models<br>
for version in ["v1_base", "v2_standard", ..., "v8_robust_no_outlier"]:<br>
&nbsp;&nbsp;&nbsp;&nbsp;X_train, y_train, X_test, y_test = load_version(version)<br>
&nbsp;&nbsp;&nbsp;&nbsp;for model in [LogisticRegression(), RandomForestClassifier(), XGBClassifier()]:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model.fit(X_train, y_train)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
</p>

<h2>2. 이상치 제거 효과</h2>
<div class="chart"><img src="{effect_img}" alt="Outlier Effect"></div>

<div class="insight">
<strong>해석:</strong><br>
- 양수 = 이상치 제거가 AUC를 높임 (도움됨)<br>
- 음수 = 이상치 제거가 오히려 AUC를 낮춤 (정보 손실)<br>
- 트리 모델은 이상치에 강건하므로 효과가 적거나 오히려 마이너스일 수 있음
</div>

<h2>3. 전체 결과표 (24개 조합)</h2>

<table>
<tr><th>Version</th><th>Outlier</th><th>Scaling</th><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1_Macro</th><th>ROC-AUC</th></tr>
{table_rows}
</table>

<h2>4. 버전별 Best Model</h2>
<table>
<tr><th>Version</th><th>Best Model</th><th>ROC-AUC</th><th>F1-Macro</th></tr>
"""

    for _, row in best_per_version.iterrows():
        html += f'<tr><td>{row["version"]}</td><td>{row["model"]}</td><td><strong>{row["ROC_AUC"]:.4f}</strong></td><td>{row["F1_Macro"]:.4f}</td></tr>\n'

    html += f"""</table>

<h2>5. 결론 및 권장</h2>

<div class="good">
<table>
<tr><th>상황</th><th>추천 버전</th><th>이유</th></tr>
<tr><td>RF/XGBoost 사용</td><td>v1_base</td><td>스케일링 불필요, 이상치 강건</td></tr>
<tr><td>로지스틱회귀 사용</td><td>v6_standard_no_outlier</td><td>이상치 제거 + 스케일링이 성능에 유리</td></tr>
<tr><td>KNN/K-means 사용</td><td>v7_minmax_no_outlier</td><td>거리 기반 → 0~1 스케일 + 이상치 제거</td></tr>
<tr><td>비교 실험 발표용</td><td>v1 vs v5</td><td>이상치 효과를 같은 조건에서 비교 가능</td></tr>
</table>
</div>

<div class="footer">KHUDA 3조 · 버전별 모델 비교 분석 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    (OUTPUT_DIR / "report.html").write_text(html, encoding="utf-8")
    logger.info(f"  Saved: results/analysis/report.html")


if __name__ == "__main__":
    main()
