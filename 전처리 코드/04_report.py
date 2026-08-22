"""
04_report.py — EDA 결과 HTML 보고서 생성

흐름: 결측치 → 이상치 → 다중공선성 → 처리 방향 요약

실행:
    python src/eda/04_report.py

출력:
    results/eda/report.html
"""

import base64
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/eda")

FEATURE_KR = {
    "hr_style": "각질 (선행형/추입형 스타일)",
    "hr_style_sd": "각질 일관성",
    "jkhr_winrate": "기수+마필 조합 승률",
    "jkhr_starts": "기수+마필 조합 출전 횟수",
    "race_style_mean": "경주 내 평균 각질",
    "race_style_sd": "경주 내 각질 편차",
    "race_front_ratio": "경주 내 선행마 비율",
    "style_vs_race": "내 각질 - 경주 평균",
    "hr_prev_rating": "직전 경주 레이팅",
    "rating": "마필 레이팅",
    "rating__z": "레이팅 (경주 내 z점수)",
    "rating__pr": "레이팅 (경주 내 백분위)",
    "hr_winrate": "마필 통산 승률",
    "hr_plcrate": "마필 통산 입상률",
    "hr_resid": "마필 평균 잔차",
    "hr_starts": "마필 통산 출주 횟수",
    "hr_last_finpct": "직전 착순 백분위",
    "hr_last_poppct": "직전 인기 백분위",
    "hr_last_ord": "직전 착순",
    "hr_last_resid": "직전 잔차",
    "hr_last_dist": "직전 경주 거리 (m)",
    "hr_last_wg": "직전 마체중",
    "hr_rest_days": "휴양일수",
    "hr_rest_days__z": "휴양일수 (z점수)",
    "hr_dist_winrate": "같은 거리 승률",
    "hr_dist_starts": "같은 거리 출주 횟수",
    "hr_dist_chg": "직전 대비 거리 변화 (m)",
    "wgBudam_chg": "부담중량 증감",
    "hr_winrate__z": "승률 (z점수)",
    "hr_winrate__pr": "승률 (백분위)",
    "hr_resid__z": "잔차 (z점수)",
    "age": "마필 연령 (세)",
    "age__z": "연령 (z점수)",
    "age__pr": "연령 (백분위)",
    "wg": "마체중 (kg)",
    "wg__z": "마체중 (z점수)",
    "wg_diff": "직전 대비 체중 증감 (kg)",
    "wg_diff__z": "체중 증감 (z점수)",
    "wgBudam": "부담중량 (kg)",
    "wgBudam__z": "부담중량 (z점수)",
    "chaksun1": "1착 상금 (경주 격)",
    "chaksun2": "2착 상금",
    "jk_winrate": "기수 승률",
    "jk_plcrate": "기수 입상률",
    "jk_starts": "기수 통산 기승 수",
    "jk_resid": "기수 평균 잔차",
    "tr_winrate": "조교사 승률",
    "tr_plcrate": "조교사 입상률",
    "tr_starts": "조교사 통산 출전 수",
    "tr_resid": "조교사 평균 잔차",
    "ow_starts": "마주 통산 출전 수",
    "ow_winrate": "마주 승률",
    "ow_plcrate": "마주 입상률",
    "ow_resid": "마주 평균 잔차",
    "train_runs_14": "14일 훈련 주행 횟수",
    "train_runs_14__z": "14일 훈련량 (z점수)",
    "train_days_14": "14일 훈련 일수",
    "train_sec_14": "14일 훈련 시간 합계 (초)",
    "bleed__pr": "폐출혈 (백분위)",
    "bleed": "폐출혈 횟수",
    "is_debut": "첫 출주 (데뷔전) 여부",
    "is_new_horse": "신마 여부",
    "is_front": "선행형 여부",
    "birthday": "생년월일 (YYYYMMDD)",
    "clinic_30d": "경주 전 30일 진료 건수",
    "n_run": "출주두수 (이번 경주 출전 말 수)",
    "dusu": "발표 출주두수",
    "rcDist": "경주 거리 (m)",
    "ilsu": "경주일 일련번호",
    "waterRate": "주로 함수율 (%)",
    "spRating": "경주 레이팅 하한",
    "stRating": "경주 레이팅 상한",
    "start_delay": "발주 지연 이력",
    "tool_n": "착용 장구 개수",
    "ill_n": "과거 진료 이력 건수",
    "pace_conflict": "선행마 경합 (초반 경쟁 정도)",
    "race_front_n": "경주 내 선행마 수",
    "tr_multi": "같은 조교사 마필 수 (경주 내)",
    "buga1": "부가상금 1",
    "wgJk": "기수 중량 보정",
    "hr_style_n": "각질 산출에 쓴 과거 경주 수",
}


def get_kr(col: str) -> str:
    return FEATURE_KR.get(col, col)


def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


CSS = """
<style>
    body { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; line-height: 1.8; font-size: 15px; }
    h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
    h2 { color: #283593; margin-top: 40px; border-left: 4px solid #3f51b5; padding-left: 12px; }
    h3 { color: #37474f; }
    .summary { background: #e8eaf6; border-radius: 8px; padding: 20px; margin: 20px 0; }
    .insight { background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 18px; margin: 15px 0; }
    .good { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 14px 18px; margin: 15px 0; }
    .warn { background: #fce4ec; border-left: 4px solid #e91e63; padding: 14px 18px; margin: 15px 0; }
    .data-table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
    .data-table th { background: #3f51b5; color: white; padding: 10px 8px; text-align: left; }
    .data-table td { padding: 8px; border-bottom: 1px solid #e0e0e0; }
    .data-table tr:hover { background: #e8eaf6; }
    .chart { text-align: center; margin: 20px 0; }
    .chart img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .explain { background: #f5f5f5; padding: 15px; border-radius: 6px; margin: 10px 0; }
    .code-block { background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px; margin: 12px 0; overflow-x: auto; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px; line-height: 1.6; color: #333; }
    .code-label { font-size: 12px; color: #666; margin-bottom: 4px; }
    .footer { margin-top: 40px; padding-top: 15px; border-top: 1px solid #ccc; color: #777; font-size: 12px; text-align: center; }
</style>
"""


def build_report() -> str:
    missing = pd.read_csv(OUTPUT_DIR / "missing_report.csv") if (OUTPUT_DIR / "missing_report.csv").exists() else None
    pattern = pd.read_csv(OUTPUT_DIR / "missing_pattern_report.csv") if (OUTPUT_DIR / "missing_pattern_report.csv").exists() else None
    pairs = pd.read_csv(OUTPUT_DIR / "high_correlation_pairs.csv") if (OUTPUT_DIR / "high_correlation_pairs.csv").exists() else None
    outlier_df = pd.read_csv(OUTPUT_DIR / "outlier_summary.csv") if (OUTPUT_DIR / "outlier_summary.csv").exists() else None

    heatmap_top40 = img_to_base64(OUTPUT_DIR / "correlation_heatmap_top40.png")
    boxplot_img = img_to_base64(OUTPUT_DIR / "outlier_boxplots.png")
    barplot_img = img_to_base64(OUTPUT_DIR / "outlier_barplot.png")

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>EDA 결과 보고서</title>{CSS}</head><body>

<h1>EDA 결과 보고서</h1>

<div class="summary">
<h3>이 보고서의 흐름</h3>
<ol>
<li><strong>결측치</strong> — 빈칸이 얼마나, 왜 있는지 (1~2장)</li>
<li><strong>이상치</strong> — 극단적으로 튀는 값이 어디에 있는지 (3장)</li>
<li><strong>다중공선성</strong> — 중복 정보를 가진 컬럼이 있는지 (4장)</li>
<li><strong>처리 방향</strong> — 위 문제들을 어떻게 해결했는지 (5장)</li>
</ol>
<p>대상: 서울 경마장 데이터 (~32,888행 × 156컬럼)</p>
</div>

<!-- ===== 1장: 결측치 ===== -->
<h2>1. 결측치(빈칸) 현황</h2>

<div class="explain">
<strong>결측치란?</strong> 데이터에서 값이 비어있는 칸. 모델에 넣기 전에 채우거나 해당 컬럼을 제거해야 합니다.
</div>
"""

    if missing is not None:
        no_miss = (missing["missing_pct"] == 0).sum()
        low = ((missing["missing_pct"] > 0) & (missing["missing_pct"] <= 10)).sum()
        mid = ((missing["missing_pct"] > 10) & (missing["missing_pct"] <= 30)).sum()
        high = (missing["missing_pct"] > 30).sum()

        html += f"""
<p class="code-label">Core code:</p>
<div class="code-block">
<pre>missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
report = pd.DataFrame({{"column": df.columns, "missing_pct": missing_pct}})
report = report.sort_values("missing_pct", ascending=False)</pre>
</div>

<table class="data-table">
<tr><th>구간</th><th>컬럼 수</th><th>의미</th></tr>
<tr><td>결측 0%</td><td><strong>{no_miss}개</strong></td><td>빈칸 없음</td></tr>
<tr><td>0~10%</td><td>{low}개</td><td>약간 빈칸. 채워서 사용</td></tr>
<tr><td>10~30%</td><td>{mid}개</td><td>원인 파악 필요</td></tr>
<tr><td>30% 이상</td><td><strong>{high}개</strong></td><td>특수 처리 또는 제거</td></tr>
</table>

<h3>결측 많은 컬럼 Top 10</h3>
<table class="data-table">
<tr><th>#</th><th>컬럼</th><th>설명</th><th>결측률</th></tr>
"""
        for i, (_, row) in enumerate(missing[missing["missing_pct"] > 0].head(10).iterrows(), 1):
            html += f'<tr><td>{i}</td><td>{row["column"]}</td><td>{get_kr(row["column"])}</td><td><strong>{row["missing_pct"]:.1f}%</strong></td></tr>\n'
        html += "</table>\n"

    # ===== 2장: 결측 패턴 =====
    html += """
<!-- ===== 2장: 결측 패턴 ===== -->
<h2>2. 왜 비어있나? (구조적 vs 랜덤)</h2>

<p class="code-label">Core code:</p>
<div class="code-block">
<pre>for col in high_missing_cols:
    for group_col in ["is_debut", "rank", "ageCond"]:
        group_missing = df.groupby(group_col)[col].apply(
            lambda x: x.isnull().mean() * 100
        )
        if group_missing.max() >= 90:
            classification = "structural"
            break
    else:
        classification = "random"</pre>
</div>

<div class="explain">
<strong>구조적 결측:</strong> 원래 값이 존재할 수 없어서 빈 것 (예: 데뷔전 말은 과거 성적이 없음)<br>
<strong>랜덤 결측:</strong> 있어야 하는데 우연히 빠진 것
</div>
"""

    if pattern is not None:
        structural = pattern[pattern["classification"] == "structural"]
        random_m = pattern[pattern["classification"] == "random"]

        if len(structural) > 0:
            html += f"""<h3>구조적 결측 ({len(structural)}개)</h3>
<table class="data-table">
<tr><th>컬럼</th><th>설명</th><th>결측률</th><th>집중 그룹</th></tr>
"""
            for _, row in structural.head(10).iterrows():
                html += f'<tr><td>{row["column"]}</td><td>{get_kr(row["column"])}</td><td>{row["overall_missing_pct"]:.1f}%</td><td>{row["concentrated_group"]} ({row["group_missing_pct"]:.0f}%)</td></tr>\n'
            html += "</table>\n"

        if len(random_m) > 0:
            html += f"""<h3>랜덤 결측 ({len(random_m)}개)</h3>
<table class="data-table">
<tr><th>컬럼</th><th>설명</th><th>결측률</th></tr>
"""
            for _, row in random_m.iterrows():
                html += f'<tr><td>{row["column"]}</td><td>{get_kr(row["column"])}</td><td>{row["overall_missing_pct"]:.1f}%</td></tr>\n'
            html += "</table>\n"

    # ===== 3장: 이상치 =====
    html += """
<!-- ===== 3장: 이상치 ===== -->
<h2>3. 이상치(극단값) 탐지</h2>

<p class="code-label">Core code:</p>
<div class="code-block">
<pre>Q1 = train[col].quantile(0.25)
Q3 = train[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

is_outlier = (df[col] < lower) | (df[col] > upper)
# outlier rate > 30% -> skip (natural wide distribution)</pre>
</div>

<div class="explain">
<strong>이상치란?</strong> 다른 데이터와 동떨어진 극단적으로 크거나 작은 값.<br>
예) 평균 휴양일 42일인데 868일 쉰 말.<br><br>
<strong>탐지 방법 (IQR):</strong> Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과 = 이상치
</div>

<div class="insight">
<strong>경마 데이터 주의점:</strong><br>
• 1착 상금은 국1등급(8.8억)과 국6등급(1,375만)이 64배 차이 → 이상치 아님, 등급 차이<br>
• 이상치 비율 30% 초과 피처는 분포 자체가 넓은 것으로 판단하여 제거 대상에서 제외<br>
• 이상치 제거/미제거 두 버전을 모두 제공 → 모델러가 비교 실험
</div>
"""

    if outlier_df is not None:
        if boxplot_img:
            html += f"""<h3>Boxplot (상위 12개 피처)</h3>
<div class="chart"><img src="{boxplot_img}" alt="Outlier Boxplots"></div>
<div class="explain">빨간 점선 = IQR 경계. 경계 밖의 점들이 이상치입니다.</div>
"""
        if barplot_img:
            html += f"""<h3>피처별 이상치 비율</h3>
<div class="chart"><img src="{barplot_img}" alt="Outlier Rate"></div>
"""

        html += """<h3>이상치 상세 (상위 15개)</h3>
<table class="data-table">
<tr><th>#</th><th>피처</th><th>설명</th><th>이상치 수</th><th>비율</th><th>제거?</th></tr>
"""
        for i, (_, row) in enumerate(outlier_df.head(15).iterrows(), 1):
            removed_txt = "O" if row["removed"] else "X (분포 넓음)"
            html += f'<tr><td>{i}</td><td>{row["feature"]}</td><td>{get_kr(row["feature"])}</td><td>{row["n_outliers"]:,.0f}</td><td>{row["pct"]:.1f}%</td><td>{removed_txt}</td></tr>\n'
        html += "</table>\n"

    # ===== 4장: 다중공선성 =====
    html += """
<!-- ===== 4장: 다중공선성 ===== -->
<h2>4. 중복 정보 (다중공선성)</h2>

<p class="code-label">Core code:</p>
<div class="code-block">
<pre>corr = df[numeric_features].corr()

# Extract pairs with |r| >= 0.8
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if abs(corr.iloc[i, j]) >= 0.8:
            high_pairs.append((cols[i], cols[j], corr.iloc[i, j]))

sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1)</pre>
</div>

<div class="explain">
<strong>다중공선성이란?</strong> 두 컬럼이 거의 같은 정보를 담고 있어서 하나만 남겨야 하는 것.<br>
예) "1착 상금"과 "2착 상금"은 항상 비례 → 하나만 남기기.<br>
<strong>기준:</strong> 상관계수 |r| >= 0.8
</div>
"""

    if heatmap_top40:
        html += f"""<h3>상관관계 히트맵 (상위 40개 피처)</h3>
<div class="chart"><img src="{heatmap_top40}" alt="Heatmap"></div>
"""

    if pairs is not None and len(pairs) > 0:
        html += f"""<h3>고상관 쌍 (총 {len(pairs)}쌍, 상위 15개)</h3>
<table class="data-table">
<tr><th>#</th><th>컬럼 A</th><th>컬럼 B</th><th>상관계수</th><th>제거 후보</th><th>이유</th></tr>
"""
        for i, (_, row) in enumerate(pairs.head(15).iterrows(), 1):
            f1, f2 = row["feature_1"], row["feature_2"]
            r = row["correlation"]
            more_missing = row["more_missing_side"]
            if more_missing != "same":
                drop, reason = more_missing, "결측 많음"
            elif "__z" in f2 or "__pr" in f2:
                drop, reason = f2, "파생 컬럼"
            elif "__z" in f1 or "__pr" in f1:
                drop, reason = f1, "파생 컬럼"
            else:
                drop, reason = f2, "중복"
            html += f'<tr><td>{i}</td><td>{f1}</td><td>{f2}</td><td>{r:.4f}</td><td>{drop}</td><td>{reason}</td></tr>\n'
        html += "</table>\n"

    # ===== 5장: 처리 요약 =====
    html += """
<!-- ===== 5장: 처리 요약 ===== -->
<h2>5. 최종 처리 방향 요약</h2>

<div class="good">
<table class="data-table">
<tr><th>문제</th><th>처리</th><th>적용 결과</th></tr>
<tr><td>구조적 결측</td><td>0 채움 + 플래그 유지</td><td>rating_na 등으로 구분</td></tr>
<tr><td>랜덤 결측</td><td>train 중앙값 대입</td><td>valid/test에도 동일값 적용</td></tr>
<tr><td>이상치</td><td>IQR 기반 행 제거</td><td>v5~v8 = 이상치 제거 버전</td></tr>
<tr><td>다중공선성</td><td>한 쪽 컬럼 제거</td><td>134컬럼 → ~111컬럼</td></tr>
</table>
</div>

<div class="warn">
<strong>주의</strong><br>
• 모든 처리는 train set 기준으로 결정, valid/test에 동일 적용 (정보 누수 방지)<br>
• 이상치 제거/미제거 두 트랙을 비교하여 모델 성능이 나은 쪽을 채택<br>
• 트리 모델(RF, XGBoost)은 이상치에 강건하므로 v1(미제거)도 유효
</div>
"""

    html += f"""
<div class="footer">KHUDA 3조 · EDA 결과 보고서 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    return html


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("EDA HTML Report Generation")
    logger.info("=" * 60)

    html = build_report()
    (OUTPUT_DIR / "report.html").write_text(html, encoding="utf-8")
    logger.info(f"  Saved: results/eda/report.html")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
