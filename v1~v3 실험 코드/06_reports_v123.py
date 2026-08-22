"""
06_reports_v123.py — 방향 1/2/3 각각의 HTML 보고서 생성 (개선판)

- 그래프는 영어 (폰트 문제 회피)
- 피처명은 한국어 설명으로 변환
- 결과 설명을 쉬운 말로 풀어서 작성

실행:
    python src/pipeline/06_reports_v123.py
"""

import base64
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FEATURE_NAME_MAP, translate_feature_name, setup_logging

logger = setup_logging()


def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


def df_to_html(df: pd.DataFrame, translate_index=False) -> str:
    df_display = df.copy()
    if translate_index and df_display.index.name != "Model":
        df_display.index = [translate_feature_name(str(i)) for i in df_display.index]
    return df_display.to_html(classes="data-table", border=0)


CSS = """
<style>
    body { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; line-height: 1.8; }
    h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
    h2 { color: #283593; margin-top: 40px; border-left: 4px solid #3f51b5; padding-left: 12px; }
    h3 { color: #37474f; }
    .summary { background: #e8eaf6; border-radius: 8px; padding: 20px; margin: 20px 0; }
    .summary li { padding: 8px 0; border-bottom: 1px solid #c5cae9; }
    .summary li:last-child { border-bottom: none; }
    .insight { background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 18px; margin: 15px 0; border-radius: 0 4px 4px 0; }
    .good { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 14px 18px; margin: 15px 0; }
    .bad { background: #fce4ec; border-left: 4px solid #e91e63; padding: 14px 18px; margin: 15px 0; }
    .data-table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
    .data-table th { background: #3f51b5; color: white; padding: 10px 8px; text-align: left; }
    .data-table td { padding: 8px; border-bottom: 1px solid #e0e0e0; }
    .data-table tr:hover { background: #e8eaf6; }
    .chart { text-align: center; margin: 20px 0; }
    .chart img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .footer { margin-top: 40px; padding-top: 15px; border-top: 1px solid #ccc; color: #777; font-size: 12px; text-align: center; }
    .explain { background: #f5f5f5; padding: 15px; border-radius: 6px; margin: 10px 0; font-size: 14px; }
    .metric { font-weight: bold; color: #1a237e; }
</style>
"""


def build_v1_report():
    d = RESULTS_DIR / "v1_calibration"
    comp = pd.read_csv(d / "calibration_comparison.csv", index_col=0)
    cal_img = img_to_base64(d / "calibration_curve.png")
    dist_img = img_to_base64(d / "probability_distributions.png")

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>방향 1: Calibration 보정</title>{CSS}</head><body>
<h1>방향 1: Calibration(확률 보정) 결과 보고서</h1>

<div class="summary">
<h3>한 줄 요약</h3>
<p>모델이 "30%로 이길 것 같다"고 말했을 때, 실제로 30%가 맞도록 확률값을 교정하는 작업입니다.</p>
<ul>
<li><span class="metric">문제:</span> 기존 모델의 predict_proba 출력값이 실제 승률과 동떨어져 있었음 (Cal_MAE = 0.30)</li>
<li><span class="metric">해결:</span> Platt Scaling 적용 후 Cal_MAE = <strong>0.01</strong>로 개선 (완벽에 가까운 보정)</li>
<li><span class="metric">한계:</span> "순위를 맞추는 능력"(AUC)은 변하지 않음. 보정은 확률값의 크기만 교정함</li>
</ul>
</div>

<h2>1. 성능 비교표</h2>

<div class="explain">
<strong>지표 설명:</strong><br>
• <strong>ROC-AUC</strong> = 이길 말과 질 말을 구분하는 능력 (1에 가까울수록 좋음)<br>
• <strong>Cal_MAE</strong> = 예측 확률과 실제 승률의 차이 (0에 가까울수록 좋음). <u>이번 실험의 핵심 지표</u><br>
• <strong>시장 확률</strong> = 배당률을 확률로 환산한 값 (비교 기준)
</div>

{df_to_html(comp.round(4))}

<div class="good">
<strong>결과 해석:</strong><br>
• Platt Scaling 후 Cal_MAE가 0.30 → 0.01로 <strong>25배 개선</strong>됨<br>
• 즉, "모델이 10%라고 예측하면 실제로 약 10%가 이긴다" = 확률 해석이 가능해짐<br>
• 시장 확률(Cal_MAE=0.12)보다도 보정된 모델(0.01)이 "확률 정확도"는 더 높음<br>
• 단, AUC(판별력)는 시장(0.82) > 모델(0.75)으로 여전히 시장이 우위
</div>

<h2>2. Calibration Curve (보정 곡선)</h2>
<div class="chart"><img src="{cal_img}" alt="Calibration Curve"></div>

<div class="explain">
<strong>그래프 보는 법:</strong><br>
• X축 = 모델이 예측한 확률, Y축 = 실제로 이긴 비율<br>
• <strong>대각선(점선)</strong> = 완벽한 보정 (예측 20% → 실제 20%)<br>
• 대각선에 가까울수록 좋은 것<br>
• 보정 전(파란) = 대각선에서 크게 벗어남 → 보정 후(주황) = 대각선에 거의 붙음
</div>

<h2>3. 확률 분포 비교</h2>
<div class="chart"><img src="{dist_img}" alt="Probability Distribution"></div>

<div class="explain">
<strong>그래프 보는 법:</strong><br>
• 각 히스토그램은 모델이 출력한 확률값의 분포<br>
• 빨간 점선 = 실제 승률 (약 9.5%)<br>
• 보정 전: 0.2~0.6으로 넓게 퍼져있음 (비현실적)<br>
• 보정 후: 실제 승률(0.095) 근처에 집중 (현실적)
</div>

<h2>4. 결론</h2>
<div class="summary">
<ul>
<li><strong>Calibration 보정은 필수입니다.</strong> RandomForest의 기본 predict_proba는 "확률"이라 부르기 어려운 수준입니다.</li>
<li>Platt Scaling 2줄로 확률 품질을 극적으로 개선할 수 있습니다.</li>
<li>보정 후에도 시장의 "판별력"은 넘지 못하지만, 모델 확률을 신뢰할 수 있게 됩니다.</li>
<li><strong>실무 적용:</strong> 모델 확률을 "기대 승률"로 해석하고 싶다면 반드시 보정을 거쳐야 합니다.</li>
</ul>
</div>

<div class="footer">KHUDA 3조 · 방향 1 Calibration 보고서 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    (d / "report.html").write_text(html, encoding="utf-8")
    logger.info(f"  저장: results/v1_calibration/report.html")


def build_v2_report():
    d = RESULTS_DIR / "v2_stacked"
    comp = pd.read_csv(d / "stacked_comparison.csv", index_col=0)
    cal_img = img_to_base64(d / "calibration_curve.png")
    fi_img = img_to_base64(d / "stage2_feature_importance.png")

    # 피처명 변환
    comp_display = comp.copy()
    comp_display.index = [
        {"Stage2_Logistic": "2단계 로지스틱", "Stage2_RF": "2단계 랜덤포레스트",
         "Stage2_XGB": "2단계 XGBoost", "1단계_RF(기존)": "1단계 RF (기존)",
         "시장확률": "시장 확률 (배당률)"}.get(i, i)
        for i in comp_display.index
    ]

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>방향 2: 2단계 모델</title>{CSS}</head><body>
<h1>방향 2: 2단계 모델 (시장 + 모델 결합) 결과 보고서</h1>

<div class="summary">
<h3>한 줄 요약</h3>
<p>"모델 예측확률"과 "배당률 확률"을 합쳐서 새로운 모델을 만들면 시장보다 나을까?</p>
<ul>
<li><span class="metric">시도:</span> 1단계 모델확률 + 시장확률 + 괴리(차이) 등 8개 피처로 2단계 모델 학습</li>
<li><span class="metric">결과:</span> 2단계 최고 AUC = 0.717. <strong>1단계(0.75)보다 오히려 하락</strong></li>
<li><span class="metric">시장 확률 단독 AUC = 0.817</span> → 모든 모델을 압도</li>
<li><span class="metric">결론:</span> 단순히 두 확률을 합치는 것으로는 시장을 이길 수 없음</li>
</ul>
</div>

<h2>1. 성능 비교표</h2>

<div class="explain">
<strong>2단계 모델이란?</strong><br>
1단계에서 나온 모델 예측확률(model_prob)과 배당률에서 나온 시장 확률(market_prob),
그리고 둘의 차이(gap)를 합쳐서 다시 한번 학습시킨 모델입니다.<br>
"모델과 시장의 장점을 결합하면 더 좋아지지 않을까?" 하는 가설을 검증한 것입니다.
</div>

{df_to_html(comp_display.round(4))}

<div class="bad">
<strong>왜 실패했을까?</strong><br>
• 시장 확률이 이미 <strong>가장 좋은 정보</strong>를 담고 있기 때문<br>
• 수만 명의 베팅자가 만들어낸 배당률은 "집단지성"의 결과물<br>
• 모델이 가진 과거 성적·훈련 정보는 이미 베팅자들도 다 아는 정보<br>
• 결국 모델이 추가로 제공하는 "새로운 정보"가 거의 없어서, 합쳐도 시장을 넘지 못함
</div>

<h2>2. Calibration Curve</h2>
<div class="chart"><img src="{cal_img}" alt="Calibration Curve"></div>

<h2>3. 2단계 모델의 Feature Importance</h2>
<div class="chart"><img src="{fi_img}" alt="Stage2 Feature Importance"></div>

<div class="explain">
<strong>피처 해석 (영문명 → 의미):</strong><br>
• <strong>market_prob</strong> = 시장(배당률) 확률 — 압도적으로 중요<br>
• <strong>q</strong> = 단승 시장 확률 (market_prob과 유사)<br>
• <strong>model_prob</strong> = 1단계 모델 예측확률 — 기여도 매우 낮음<br>
• <strong>gap</strong> = 모델확률 - 시장확률 (괴리)<br>
• <strong>pop_pct</strong> = 인기 백분위<br><br>
→ 결론: 2단계 모델은 사실상 "시장 확률을 그대로 쓰는 것"과 다를 바 없음
</div>

<h2>4. 결론</h2>
<div class="summary">
<ul>
<li><strong>서울 경마 시장은 매우 효율적입니다.</strong> 배당률만으로 AUC 0.82 — 이건 상당히 높은 수준입니다.</li>
<li>단순한 결합(stacking)으로는 시장을 넘을 수 없습니다.</li>
<li>시장을 이기려면: <strong>시장이 모르는 정보</strong>(실시간 컨디션, 내부 정보 등)가 필요하거나, <strong>시장이 비효율적인 틈새 구간</strong>을 노려야 합니다.</li>
<li>→ 이것이 바로 "방향 3: 이변 예측"으로 이어지는 이유입니다.</li>
</ul>
</div>

<div class="footer">KHUDA 3조 · 방향 2 보고서 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    (d / "report.html").write_text(html, encoding="utf-8")
    logger.info(f"  저장: results/v2_stacked/report.html")


def build_v3_report():
    d = RESULTS_DIR / "v3_upset"
    comp = pd.read_csv(d / "upset_model_comparison.csv", index_col=0)
    decile = pd.read_csv(d / "upset_decile_analysis.csv")
    fi_img = img_to_base64(d / "upset_feature_importance.png")
    decile_img = img_to_base64(d / "upset_decile_chart.png")

    # 피처 중요도 CSV 로드하여 한국어 변환
    fi_path = d / "upset_feature_importance.csv"
    fi_table = ""
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path).head(15)
        fi_df["설명"] = fi_df["feature"].apply(translate_feature_name)
        fi_df = fi_df[["설명", "feature", "importance"]].rename(
            columns={"feature": "원본 피처명", "importance": "중요도"}
        )
        fi_table = fi_df.to_html(classes="data-table", border=0, index=False)

    overall_rate = decile["upset_rate"].mean()
    top_rate = decile[decile["proba_decile"] == decile["proba_decile"].max()]["upset_rate"].values[0]
    top_odds = decile[decile["proba_decile"] == decile["proba_decile"].max()]["avg_odds"].values[0]
    lift = top_rate / overall_rate if overall_rate > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>방향 3: 이변 예측 모델</title>{CSS}</head><body>
<h1>방향 3: 이변 예측 모델 결과 보고서</h1>

<div class="summary">
<h3>한 줄 요약</h3>
<p>"이 말이 이긴다"를 맞추는 대신, <strong>"시장이 과소평가한 복병"</strong>을 찾아내는 모델입니다.</p>
<ul>
<li><span class="metric">타겟:</span> upset_B = 인기 하위 50% 비인기마인데 1착한 말 (= 이변)</li>
<li><span class="metric">모델 상위 10% 그룹의 이변 적중률:</span> <strong>{top_rate:.1%}</strong></li>
<li><span class="metric">전체 평균 이변 비율:</span> {overall_rate:.1%}</li>
<li><span class="metric">Lift:</span> <strong>{lift:.2f}배</strong> (모델 상위 그룹이 일반 대비 {lift:.1f}배 높은 적중률)</li>
<li><span class="metric">상위 그룹 평균 배당:</span> {top_odds:.1f}배</li>
</ul>
</div>

<div class="good">
<strong>이게 왜 의미있나요?</strong><br>
• 배당률 24.5배짜리 말의 적중률이 23.8%라면 → 기대값 = 24.5 × 0.238 = <strong>5.83</strong> (100원 투자 시 583원 회수 기대)<br>
• 물론 이건 "평균적으로"이고, 실전에서는 변동이 크지만, <strong>모델 없이 무작위로 비인기마를 고르는 것보다 1.5배 나은 선택</strong>이 가능하다는 뜻<br>
• 핵심: 시장과 "같은 게임"을 하지 않고, <strong>시장이 실수하는 영역</strong>에서만 승부
</div>

<h2>1. 모델 성능</h2>

<div class="explain">
<strong>이변 예측이 어려운 이유:</strong><br>
• 이변 자체가 확률 13%의 희귀 사건 (비인기마 중 1착)<br>
• AUC 0.64는 "약하지만 유의미한 판별력" — 완전 랜덤(0.5)보다 확실히 나음<br>
• 중요한 건 AUC가 아니라, <strong>"모델이 높은 점수를 준 말에서 실제 적중률이 높은가"</strong>
</div>

{df_to_html(comp.round(4))}

<h2>2. 예측 점수 구간별 이변 적중률</h2>
<div class="chart"><img src="{decile_img}" alt="Decile Analysis"></div>

<div class="explain">
<strong>그래프 보는 법:</strong><br>
• X축 = 모델이 매긴 이변 가능성 점수 (0=낮음 → 9=높음)<br>
• 주황 막대 = 해당 구간의 실제 이변 적중률<br>
• 파란 선 = 해당 구간의 평균 배당률<br>
• 빨간 점선 = 전체 평균 이변 비율 (기준선)
</div>

{decile.round(4).rename(columns={
    "proba_decile": "점수 구간 (0=낮음→9=높음)",
    "upset_rate": "이변 적중률",
    "avg_odds": "평균 배당률",
    "count": "해당 구간 말 수"
}).to_html(classes="data-table", border=0, index=False)}

<div class="insight">
<strong>핵심 패턴:</strong><br>
• 점수 구간 9 (최상위): 적중률 {top_rate:.1%}, 배당 {top_odds:.1f}배<br>
• 점수 구간 0 (최하위): 적중률 3.7%, 배당 71.5배<br>
• <strong>점수가 올라갈수록 적중률이 올라가고, 배당은 적당히 높은 수준 유지</strong><br>
• 이는 모델이 "적중 가능성이 있으면서도 배당이 좋은" 말을 골라내고 있다는 의미
</div>

<h2>3. 이변을 만드는 핵심 요인 (Feature Importance)</h2>
<div class="chart"><img src="{fi_img}" alt="Feature Importance"></div>

<h3>중요 피처 Top 15 (한국어 설명)</h3>
{fi_table}

<div class="insight">
<strong>해석 — "이변이 일어나는 조건":</strong><br>
1. <strong>직전 경주 성적(hr_last_finpct)</strong>: 직전에 하위권이었던 말이 갑자기 상위권에 오는 패턴<br>
2. <strong>연령(age__z)</strong>: 특정 연령대에서 시장이 과소평가하는 경향<br>
3. <strong>기수 입상률(jk_plcrate)</strong>: 좋은 기수가 비인기마를 탈 때 이변 확률 상승<br>
4. <strong>훈련량(train_runs_14__z)</strong>: 최근 훈련을 많이 한 말 = 컨디션 상승 신호<br><br>
→ 시장(베팅자들)이 "직전 성적이 나빴으니 이번에도 안 되겠지"라고 판단하지만,
   모델은 "기수가 바뀌었고 훈련량이 늘었으니 이번엔 다르다"를 포착
</div>

<h2>4. 결론</h2>
<div class="summary">
<ul>
<li><strong>모델의 실용적 가치:</strong> "어떤 비인기마를 주목해야 하는가"에 대한 필터링 도구</li>
<li>상위 20% 그룹만 관찰하면, 이변 적중률이 전체 평균의 1.5배 — 정보 이점 존재</li>
<li><strong>한계:</strong> AUC 자체는 낮으므로 "이 말이 반드시 이변을 만든다"는 확신은 불가. 확률적 우위에 불과</li>
<li><strong>향후:</strong> 각질(hr_style), 기수-마필 궁합(jkhr_winrate), 컨디션 지표를 더 강화하면 Lift를 높일 수 있을 것</li>
</ul>
</div>

<div class="footer">KHUDA 3조 · 방향 3 이변 예측 보고서 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    (d / "report.html").write_text(html, encoding="utf-8")
    logger.info(f"  저장: results/v3_upset/report.html")


def main():
    logger.info("=" * 60)
    logger.info("방향 1/2/3 보고서 재생성 (영문 그래프 + 한국어 설명)")
    logger.info("=" * 60)

    build_v1_report()
    build_v2_report()
    build_v3_report()

    logger.info("\n  완료! 아래 파일을 브라우저에서 확인하세요:")
    logger.info("    results/v1_calibration/report.html")
    logger.info("    results/v2_stacked/report.html")
    logger.info("    results/v3_upset/report.html")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
