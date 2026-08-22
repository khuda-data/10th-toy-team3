"""
09_preprocessing_report.py — 배당률 포함 전처리 보고서

08_full_pipeline.py에서 수행한 전처리 과정을 정리한 HTML 보고서.

실행:
    python src/upset_with_odds/09_preprocessing_report.py

출력:
    results/upset_with_odds_v2/preprocessing_report.html
"""

import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds_v2")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>전처리 보고서 (배당률 포함 버전)</title>
<style>
    body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; line-height: 1.8; font-size: 15px; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 40px; border-left: 4px solid #3f51b5; padding-left: 12px; }}
    h3 {{ color: #37474f; margin-top: 25px; }}
    .summary {{ background: #e8eaf6; border-radius: 8px; padding: 20px; margin: 20px 0; }}
    .good {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 14px 18px; margin: 15px 0; }}
    .insight {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 18px; margin: 15px 0; }}
    .warn {{ background: #fce4ec; border-left: 4px solid #e91e63; padding: 14px 18px; margin: 15px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
    th {{ background: #3f51b5; color: white; padding: 9px; text-align: left; }}
    td {{ padding: 8px; border-bottom: 1px solid #e0e0e0; }}
    tr:hover {{ background: #e8eaf6; }}
    .code-block {{ background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px; margin: 12px 0; font-family: Consolas, monospace; font-size: 13px; line-height: 1.6; overflow-x: auto; }}
    .step-num {{ display: inline-block; background: #3f51b5; color: white; width: 28px; height: 28px; border-radius: 50%; text-align: center; line-height: 28px; font-size: 14px; font-weight: bold; margin-right: 8px; }}
    .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ccc; color: #777; font-size: 12px; text-align: center; }}
</style></head><body>

<h1>데이터 전처리 보고서 (배당률 포함 버전)</h1>

<div class="summary">
<h3>이 버전의 특징</h3>
<p>기존 전처리(배당률 제외)와 달리, <strong>배당률 대표 피처(q)를 포함</strong>시킨 버전입니다.<br>
이변(비인기마 1착) 예측 모델에 사용하기 위한 전처리입니다.</p>

<table>
<tr><th>단계</th><th>처리</th><th>결과</th></tr>
<tr><td><span class="step-num">1</span></td><td>서울 + 비인기마 필터링</td><td>56,648행 → ~17,000행</td></tr>
<tr><td><span class="step-num">2</span></td><td>결측치 처리</td><td>구조적 → 0, 랜덤 → train 중앙값</td></tr>
<tr><td><span class="step-num">3</span></td><td>다중공선성 제거</td><td>배당률 14개→q만 + 고상관 23개 제거</td></tr>
<tr><td><span class="step-num">4</span></td><td>범주형 인코딩</td><td>13개 문자 컬럼 → LabelEncoder</td></tr>
<tr><td><span class="step-num">5</span></td><td>스케일링</td><td>StandardScaler (train 기준)</td></tr>
<tr><td><span class="step-num">6</span></td><td>데이터 분할</td><td>시간순 6:2:2</td></tr>
<tr><td><span class="step-num">7</span></td><td>타겟 생성</td><td>upset = (비인기마) & (1착) → 1</td></tr>
</table>
</div>

<!-- ===== 1. 필터링 ===== -->
<h2><span class="step-num">1</span> 데이터 필터링</h2>

<h3>1-1. 서울 경마장만</h3>
<table>
<tr><th>구분</th><th>행 수</th></tr>
<tr><td>원본 (final.csv)</td><td>56,648</td></tr>
<tr><td>서울만</td><td>32,888</td></tr>
</table>

<h3>1-2. 비인기마만 (pop_pct >= 0.5)</h3>

<div class="insight">
<strong>왜 비인기마만?</strong><br>
- 타겟이 "이변"(비인기마가 1착)이므로, 인기마는 이변 대상 자체가 아님<br>
- 인기마 행을 포함하면 타겟 비율이 ~7%로 너무 낮아짐<br>
- 비인기마만 사용하면 타겟 비율 ~14%로 불균형이 완화됨<br>
- "비인기마 중에서 진짜 이길 말을 골라내는" 명확한 문제 정의
</div>

<table>
<tr><th>구분</th><th>행 수</th><th>upset 비율</th></tr>
<tr><td>서울 전체</td><td>32,888</td><td>~7%</td></tr>
<tr><td>비인기마만 (pop_pct >= 0.5)</td><td>~17,000</td><td>~14%</td></tr>
</table>

<div class="code-block"><pre>df = pd.read_csv("final.csv", low_memory=False)
df = df[df["meet"] == "서울"]
df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)
df = df[df["pop_pct"] >= 0.5]  # longshots only</pre></div>

<!-- ===== 2. 결측치 ===== -->
<h2><span class="step-num">2</span> 결측치 처리</h2>

<table>
<tr><th>유형</th><th>처리</th><th>대상 컬럼 (대표)</th></tr>
<tr><td>구조적 결측<br>(원래 값이 없는 게 정상)</td><td><strong>0으로 채움</strong></td><td>rating, hr_winrate, hr_plcrate, hr_style, jkhr_winrate 등 26개</td></tr>
<tr><td>랜덤 결측<br>(있어야 하는데 누락)</td><td><strong>train 중앙값</strong></td><td>hr_rest_days 등 소수</td></tr>
<tr><td>범주형 결측</td><td><strong>"MISSING"</strong></td><td>weather, track 등</td></tr>
</table>

<div class="code-block"><pre># Structural: value being absent IS the information
for col in ["rating", "hr_winrate", "hr_plcrate", ...]:
    df[col] = df[col].fillna(0)

# Random: use train median (no leakage)
medians = df.loc[train_mask, remaining_cols].median()
df[remaining_cols] = df[remaining_cols].fillna(medians)</pre></div>

<!-- ===== 3. 다중공선성 ===== -->
<h2><span class="step-num">3</span> 다중공선성 (고상관 피처) 제거</h2>

<h3>3-1. 배당률 14개 → q 하나만</h3>

<div class="insight">
<strong>왜 q만 남기나?</strong><br>
- 배당률 파생 14개 컬럼은 서로 상관계수 0.9 이상 (거의 같은 정보)<br>
- q = 오버라운드 제거 후 정규화된 암묵적 확률 (경주 내 합=1)<br>
- 가장 "깨끗한" 형태의 시장 확률이므로 대표로 선정
</div>

<table>
<tr><th>남긴 것</th><th>제거한 것 (13개)</th></tr>
<tr><td><strong>q</strong> (시장 확률)</td><td>winOdds, plcOdds, p_raw, logit_q, log_q, pop_rank, is_fav, book_sum, takeout, pl_harville, pl_disc, q_plc</td></tr>
</table>

<h3>3-2. 기존 피처 고상관 쌍 제거 (23개)</h3>

<table>
<tr><th>제거</th><th>남김</th><th>이유</th></tr>
<tr><td>chaksun2~5</td><td>chaksun1</td><td>비례 관계 (상관 0.99+)</td></tr>
<tr><td>buga2, buga3</td><td>buga1</td><td>동일 비례</td></tr>
<tr><td>dusu</td><td>n_run</td><td>거의 동일 값</td></tr>
<tr><td>hr_prev_rating</td><td>rating</td><td>상관 0.99 + 결측 더 많음</td></tr>
<tr><td>hr_last_finpct</td><td>hr_last_ord</td><td>같은 정보 다른 표현</td></tr>
<tr><td>hr_last_wg</td><td>wg</td><td>체중 거의 안 변함</td></tr>
<tr><td>__pr 컬럼 9개</td><td>__z 컬럼</td><td>z점수와 백분위 중복</td></tr>
<tr><td>기타 3개</td><td>-</td><td>상관 0.97+</td></tr>
</table>

<div class="code-block"><pre># Drop odds (keep only q)
odds_drop = ["winOdds", "plcOdds", "p_raw", "logit_q", ...]
df = df.drop(columns=odds_drop)

# Drop high-correlation pairs
corr_drop = ["chaksun2", "chaksun3", ..., "hr_last_wg", ...]
df = df.drop(columns=corr_drop)</pre></div>

<!-- ===== 4. 인코딩 ===== -->
<h2><span class="step-num">4</span> 범주형 인코딩 (문자 → 숫자)</h2>

<div class="insight">
모델은 숫자만 입력받을 수 있으므로, 문자형 컬럼을 정수로 변환합니다.<br>
<strong>LabelEncoder</strong>: 각 고유값에 정수 부여. 트리 모델에 적합 (순서 무관).
</div>

<table>
<tr><th>컬럼</th><th>예시 값</th><th>변환 후</th></tr>
<tr><td>rank</td><td>"국6등급"</td><td>→ 5</td></tr>
<tr><td>track</td><td>"건조 (3%)"</td><td>→ 0</td></tr>
<tr><td>weather</td><td>"맑음"</td><td>→ 3</td></tr>
<tr><td>sex</td><td>"수"</td><td>→ 2</td></tr>
<tr><td>rating_na</td><td>"True"</td><td>→ 1</td></tr>
</table>

<div class="code-block"><pre>from sklearn.preprocessing import LabelEncoder

for col in cat_cols:  # 13 columns
    df[col] = df[col].fillna("MISSING").astype(str)
    le = LabelEncoder()
    le.fit(df[col].unique())
    df[col] = le.transform(df[col])</pre></div>

<!-- ===== 5. 스케일링 ===== -->
<h2><span class="step-num">5</span> 스케일링 (StandardScaler)</h2>

<div class="insight">
<strong>StandardScaler:</strong> (값 - 평균) / 표준편차 → 평균=0, 분산=1로 변환<br>
로지스틱회귀가 피처 간 스케일 차이에 민감하므로 적용.<br>
RF는 스케일 무관하지만, 동일 데이터로 두 모델을 비교하기 위해 통일.
</div>

<table>
<tr><th>핵심 원칙</th><th>설명</th></tr>
<tr><td>train으로만 fit</td><td>평균/표준편차를 train에서 계산</td></tr>
<tr><td>전체에 transform</td><td>같은 기준을 valid/test에도 적용</td></tr>
<tr><td>누수 방지</td><td>valid/test의 통계를 쓰면 미래 정보가 학습에 섞임</td></tr>
</table>

<div class="code-block"><pre>from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.loc[train_mask, num_cols])   # train only!
df[num_cols] = scaler.transform(df[num_cols])  # apply to all</pre></div>

<!-- ===== 6. 분할 ===== -->
<h2><span class="step-num">6</span> 데이터 분할 (시간순 6:2:2)</h2>

<table>
<tr><th>Fold</th><th>비율</th><th>용도</th><th>기간 (예상)</th></tr>
<tr><td>train</td><td>60%</td><td>모델 학습</td><td>2023-08 ~ 2025-05</td></tr>
<tr><td>valid</td><td>20%</td><td>threshold 탐색</td><td>2025-05 ~ 2025-12</td></tr>
<tr><td>test</td><td>20%</td><td>최종 평가 + ROI</td><td>2025-12 ~ 2026-08</td></tr>
</table>

<div class="warn">
<strong>절대 섞으면 안 됨!</strong> 과거 데이터로 학습 → 미래 데이터로 평가.<br>
섞으면 "미래 경주 결과를 보고 과거를 예측"하는 꼴이 되어 성능이 부풀려짐.
</div>

<div class="code-block"><pre># Boundaries on rcDate — same race never splits across folds
df = df.sort_values("rcDate")
df["fold"] = assign_time_split(df, date_col="rcDate", ratios=(0.6, 0.2, 0.2))</pre></div>

<!-- ===== 7. 타겟 ===== -->
<h2><span class="step-num">7</span> 타겟(y) 정의: upset</h2>

<div class="good">
<table>
<tr><th>조건</th><th>upset</th><th>의미</th></tr>
<tr><td>비인기마(pop_pct >= 0.5) AND 1착(win=1)</td><td><strong>1</strong></td><td>이변 발생 (시장이 틀림)</td></tr>
<tr><td>그 외</td><td>0</td><td>이변 아님</td></tr>
</table>
<p>비인기마만 대상이므로, upset=1 비율은 약 14% (6~7:1 불균형).<br>
class_weight='balanced'로 불균형 대응.</p>
</div>

<!-- ===== 최종 ===== -->
<h2>전처리 전후 비교</h2>

<div class="good">
<table>
<tr><th></th><th>전처리 전 (final.csv)</th><th>전처리 후</th></tr>
<tr><td>행 수</td><td>56,648</td><td>~17,000 (서울 비인기마만)</td></tr>
<tr><td>컬럼 수</td><td>156</td><td>~90 (식별자/결과/고상관 제거 후)</td></tr>
<tr><td>배당률</td><td>14개 컬럼</td><td><strong>q 하나만 포함</strong></td></tr>
<tr><td>결측치</td><td>29개 컬럼 5%+</td><td>0 (전부 처리)</td></tr>
<tr><td>범주형</td><td>문자열 13개</td><td>정수 변환 완료</td></tr>
<tr><td>스케일</td><td>제각각</td><td>StandardScaler 적용</td></tr>
<tr><td>분할</td><td>없음</td><td>6:2:2 시간순</td></tr>
<tr><td>타겟</td><td>win (전체 9.5%)</td><td><strong>upset (비인기마 내 14%)</strong></td></tr>
</table>
</div>

<div class="insight">
<strong>이전 전처리(배당률 제외)와의 차이:</strong><br>
- <strong>q 포함:</strong> 배당률 대표 피처를 모델 입력으로 사용<br>
- <strong>비인기마만:</strong> 전체 32,888행 대신 절반만 사용<br>
- <strong>타겟 변경:</strong> win(1착) → upset(비인기마 1착)<br>
- 이유: "시장을 이기겠다"가 아니라 "시장이 놓치는 이변을 배당률과 함께 잡겠다"
</div>

<div class="footer">KHUDA 3조 · 전처리 보고서 (배당률 포함) · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    path = OUTPUT_DIR / "preprocessing_report.html"
    path.write_text(html, encoding="utf-8")
    logger.info(f"Saved: {path}")


if __name__ == "__main__":
    main()
