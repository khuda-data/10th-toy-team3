"""
05_model_report.py — 최종 결과 종합 보고서 생성

모든 파이프라인 결과를 하나의 HTML 보고서와 종합 그래프로 정리한다.
results/report.html 파일을 브라우저에서 열면 전체 결과를 확인 가능.

실행:
    python src/pipeline/05_model_report.py
"""

import sys
from pathlib import Path
from datetime import datetime
import base64

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RESULTS_DIR,
    setup_logging,
    ensure_dirs,
)

logger = setup_logging()


def img_to_base64(path: Path) -> str:
    """이미지를 base64로 인코딩하여 HTML 내 인라인 삽입용 문자열 반환."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def df_to_html(df: pd.DataFrame, caption: str = "") -> str:
    """DataFrame을 스타일링된 HTML 테이블로 변환."""
    html = df.to_html(index=True, classes="data-table", border=0)
    if caption:
        html = f"<h3>{caption}</h3>\n" + html
    return html


def build_report() -> str:
    """전체 결과를 종합하여 HTML 보고서 문자열을 생성한다."""

    # --- 데이터 로드 ---
    model_comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv", index_col=0)
    threshold = pd.read_csv(RESULTS_DIR / "threshold_tuning.csv")
    gap_valid = pd.read_csv(RESULTS_DIR / "gap_validation.csv")
    flb_overall = pd.read_csv(RESULTS_DIR / "flb_overall.csv")
    cluster_profiles = pd.read_csv(RESULTS_DIR / "cluster_profiles.csv")

    market_gap = pd.read_csv(RESULTS_DIR / "market_gap.csv")
    feature_comp = pd.read_csv(RESULTS_DIR / "feature_comparison.csv", index_col=0)

    # --- 이미지 인라인 ---
    fi_img = img_to_base64(RESULTS_DIR / "feature_importance.png")
    fd_img = img_to_base64(RESULTS_DIR / "feature_distribution.png")
    km_img = img_to_base64(RESULTS_DIR / "kmeans_elbow.png")
    gv_img = img_to_base64(RESULTS_DIR / "gap_validation.png")
    flb_img = img_to_base64(RESULTS_DIR / "flb_overall.png")
    flb_rank_img = img_to_base64(RESULTS_DIR / "flb_by_rank.png")
    flb_dist_img = img_to_base64(RESULTS_DIR / "flb_by_rcDist.png")
    flb_track_img = img_to_base64(RESULTS_DIR / "flb_by_track.png")
    flb_nrun_img = img_to_base64(RESULTS_DIR / "flb_by_n_run.png")
    flb_amt_img = img_to_base64(RESULTS_DIR / "flb_by_totalAmt.png")

    # --- 요약 통계 ---
    best_model = model_comp["ROC_AUC"].idxmax()
    best_auc = model_comp.loc[best_model, "ROC_AUC"]
    gap_mean = market_gap["gap"].mean()
    gap_std = market_gap["gap"].std()

    # --- HTML 조립 ---
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>경주마 승률 예측 모델 — 최종 보고서</title>
    <style>
        body {{
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{
            color: #1a237e;
            border-bottom: 3px solid #1a237e;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #283593;
            margin-top: 40px;
            border-left: 4px solid #3f51b5;
            padding-left: 12px;
        }}
        h3 {{
            color: #455a64;
            margin-top: 20px;
        }}
        .summary-box {{
            background: #e8eaf6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .summary-box ul {{
            list-style: none;
            padding: 0;
        }}
        .summary-box li {{
            padding: 6px 0;
            border-bottom: 1px solid #c5cae9;
        }}
        .summary-box li:last-child {{
            border-bottom: none;
        }}
        .metric {{
            font-weight: bold;
            color: #1a237e;
        }}
        .data-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 13px;
        }}
        .data-table th {{
            background: #3f51b5;
            color: white;
            padding: 10px 8px;
            text-align: left;
        }}
        .data-table td {{
            padding: 8px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .data-table tr:hover {{
            background: #e8eaf6;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .insight {{
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 12px 16px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }}
        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
            color: #777;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>

<h1>경주마 승률 예측 모델 — 최종 결과 보고서</h1>

<div class="summary-box">
    <h3>Executive Summary</h3>
    <ul>
        <li><span class="metric">최종 모델:</span> {best_model} (ROC-AUC = {best_auc:.4f})</li>
        <li><span class="metric">데이터:</span> 서울 경마장 32,888건 (train/valid/test 시간순 3분할)</li>
        <li><span class="metric">예측 대상:</span> 1착 여부 (win, 양성 비율 ~9.5%)</li>
        <li><span class="metric">괴리 평균:</span> {gap_mean:+.4f} ± {gap_std:.4f} (모델이 시장보다 전반적으로 높게 예측)</li>
        <li><span class="metric">결론:</span> 시장(배당률)이 모델보다 실제 승률 예측에 더 정확함. 모델은 calibration 개선 필요.</li>
    </ul>
</div>

<!-- ========== 2단계: 모델 비교 ========== -->
<h2>1. 모델 성능 비교</h2>
<p>세 모델(로지스틱회귀, 랜덤포레스트, XGBoost)을 test set에서 평가한 결과입니다.</p>

{df_to_html(model_comp.round(4), "")}

<div class="insight">
    <strong>해석:</strong> RandomForest가 ROC-AUC {best_auc:.4f}로 가장 높은 판별력을 보였습니다.
    로지스틱회귀는 Recall이 높지만 Precision이 극도로 낮아 실용성이 부족합니다.
    XGBoost는 Accuracy가 높지만 소수 클래스(win)를 거의 잡아내지 못합니다.
</div>

<!-- ========== 3단계: Threshold ========== -->
<h2>2. Threshold 튜닝</h2>
<p>RandomForest의 예측 임계값을 valid set에서 탐색한 결과입니다.</p>

{df_to_html(threshold.round(4), "")}

<div class="insight">
    <strong>해석:</strong> F1(Macro)는 threshold=0.5에서 가장 높았습니다.
    낮은 threshold(0.05~0.15)에서는 Recall이 올라가지만 Precision이 급락합니다.
</div>

<!-- ========== 5단계: Feature Importance ========== -->
<h2>3. Feature Importance</h2>

<div class="chart-container">
    <img src="{fi_img}" alt="Feature Importance Top 20">
</div>

{df_to_html(feature_comp.head(15).round(4), "괴리 상위 20% vs 전체 — 피처 평균 비교 (Top 15)")}

<div class="chart-container">
    <img src="{fd_img}" alt="피처 분포 비교">
</div>

<div class="insight">
    <strong>해석:</strong> 직전 경주 성적(hr_last_finpct, hr_last_poppct)과 연령(age__z)이 가장 중요한 피처입니다.
    괴리가 큰 말들은 이 피처들의 분포가 전체와 유의미하게 다릅니다.
</div>

<!-- ========== 6단계: K-means ========== -->
<h2>4. K-means 클러스터링 (괴리 상위 20%)</h2>

<div class="chart-container">
    <img src="{km_img}" alt="엘보우 + 실루엣">
</div>

{df_to_html(cluster_profiles.round(4), "군집별 피처 평균 (상위 컬럼만 표시)")}

<div class="insight">
    <strong>해석:</strong> 최적 k=2. 군집 1(71건)은 고액 상금 경주(국1~2등급)의 말들로,
    군집 0(2,099건)은 일반 경주의 말들입니다. 괴리가 큰 집단은 주로 상금 규모에 의해 구분됩니다.
</div>

<!-- ========== 7단계: 괴리 검증 ========== -->
<h2>5. 괴리 구간별 실제 승률 비교</h2>

<div class="chart-container">
    <img src="{gv_img}" alt="괴리 구간별 검증">
</div>

{df_to_html(gap_valid.round(4), "괴리(gap) 10분위별 집계")}

<div class="insight">
    <strong>핵심 발견:</strong> 모든 구간에서 시장 암묵적확률이 실제 승률과 더 가깝습니다.
    모델 예측확률은 전반적으로 과대추정(0.19~0.62)되어 있어 calibration이 필요합니다.
    시장 확률은 0.08~0.11 범위로, 실제 승률(0.05~0.11)과 매우 유사합니다.
</div>

<!-- ========== 8단계: FLB ========== -->
<h2>6. Favorite-Longshot Bias (FLB) 분석</h2>

<h3>6.1 기본 분석</h3>
<div class="chart-container">
    <img src="{flb_img}" alt="FLB Overall">
</div>

{df_to_html(flb_overall.round(4), "배당 구간별 실제 승률 vs 시장 확률")}

<div class="insight">
    <strong>해석:</strong> 인기마(1배대) bias = -0.012 (약간 과대평가), 비인기마(10배+) bias = -0.004 (약간 과대평가).
    한국 서울 경마에서는 전형적인 FLB가 매우 약하게 나타나거나 거의 없습니다.
    3~10배대에서 오히려 시장이 약간 과소평가하는 경향(양의 bias)이 있습니다.
</div>

<h3>6.2 조절변수별 FLB</h3>

<h4>경주등급별</h4>
<div class="chart-container">
    <img src="{flb_rank_img}" alt="FLB by 경주등급">
</div>

<h4>거리별</h4>
<div class="chart-container">
    <img src="{flb_dist_img}" alt="FLB by 거리">
</div>

<h4>트랙상태별</h4>
<div class="chart-container">
    <img src="{flb_track_img}" alt="FLB by 트랙상태">
</div>

<h4>출주두수별</h4>
<div class="chart-container">
    <img src="{flb_nrun_img}" alt="FLB by 출주두수">
</div>

<h4>매출규모별</h4>
<div class="chart-container">
    <img src="{flb_amt_img}" alt="FLB by 매출규모">
</div>

<!-- ========== 결론 ========== -->
<h2>7. 결론 및 시사점</h2>

<div class="summary-box">
    <ul>
        <li><strong>모델 성능:</strong> RandomForest AUC=0.75로 기본적인 판별력은 있으나, 확률 calibration이 부족합니다.</li>
        <li><strong>시장 효율성:</strong> 배당률 기반 시장확률이 모델 예측확률보다 실제 승률에 훨씬 가깝습니다.</li>
        <li><strong>FLB:</strong> 서울 경마에서 FLB는 매우 약하게 나타나며, 시장이 상당히 효율적입니다.</li>
        <li><strong>괴리 원인:</strong> 모델의 predict_proba가 well-calibrated되지 않아 확률값이 전반적으로 과대추정됩니다.</li>
        <li><strong>향후 과제:</strong> Platt Scaling / Isotonic Regression으로 calibration 보정, 또는 시장 확률을 피처로 활용하는 2단계 모델 구축.</li>
    </ul>
</div>

<div class="footer">
    <p>KHUDA 3조 · 토이프로젝트 · 생성일: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    <p>데이터: 한국마사회 공공데이터 API · 서울 경마장 2023-08~2026-08</p>
</div>

</body>
</html>"""

    return html


def main():
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("최종 보고서 생성")
    logger.info("=" * 60)

    html = build_report()

    output_path = RESULTS_DIR / "report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"  보고서 저장: {output_path}")
    logger.info(f"  브라우저에서 열기: file:///{output_path.as_posix()}")
    logger.info("\n" + "=" * 60)
    logger.info("05_model_report.py 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
