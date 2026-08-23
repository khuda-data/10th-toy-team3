"""
06_version_report.py — 전처리 버전 가이드 HTML 보고서

실행:
    python src/1_전처리/06_version_report.py

출력:
    data/versions/report.html
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VERSIONS_DIR

OUTPUT_DIR = VERSIONS_DIR


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>전처리 버전 가이드</title>
<style>
    body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; line-height: 1.8; font-size: 15px; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 35px; border-left: 4px solid #3f51b5; padding-left: 12px; }}
    .box {{ background: #e8eaf6; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    .warn {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 18px; margin: 15px 0; }}
    .good {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 14px 18px; margin: 15px 0; }}
    .bad {{ background: #fce4ec; border-left: 4px solid #e91e63; padding: 14px 18px; margin: 15px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }}
    th {{ background: #3f51b5; color: white; padding: 10px; text-align: left; }}
    td {{ padding: 10px; border-bottom: 1px solid #e0e0e0; }}
    tr:hover {{ background: #e8eaf6; }}
    .code-block {{ background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px; margin: 12px 0; overflow-x: auto; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px; line-height: 1.6; color: #333; }}
    .code-label {{ font-size: 12px; color: #666; margin-bottom: 4px; }}
    .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ccc; color: #777; font-size: 12px; text-align: center; }}
</style>
</head><body>

<h1>전처리 버전 가이드 — 팀원용</h1>

<div class="box">
<h3>이 폴더에 뭐가 있나요?</h3>
<p>같은 데이터를 <strong>8가지 방식으로 전처리</strong>한 CSV 파일입니다.<br>
이상치 포함/제거 x 스케일링 4종 = 8버전. 모델에 따라 골라서 쓰면 됩니다.</p>
</div>

<!-- ===== 1. 공통 처리 ===== -->
<h2>1. 모든 버전에 공통으로 적용된 처리</h2>

<table>
<tr><th>단계</th><th>처리 내용</th><th>예시</th></tr>
<tr><td>결측치 (구조적)</td><td>0으로 채움 + 플래그 유지</td><td>rating -> 0, rating_na=True</td></tr>
<tr><td>결측치 (랜덤)</td><td>train 중앙값으로 채움</td><td>hr_rest_days 누락 -> 비슷한 말들의 중간값</td></tr>
<tr><td>중복 피처 제거</td><td>상관계수 0.8+ 쌍에서 하나 삭제</td><td>chaksun2~5 삭제 (chaksun1만 유지)</td></tr>
<tr><td>데이터 분할</td><td>시간순 6:2:2 (train/valid/test)</td><td>과거 60% 학습, 최근 20% 평가</td></tr>
</table>

<p class="code-label">핵심 코드:</p>
<div class="code-block">
<pre># Structural missing -> fill with 0
structural_cols = ["rating", "hr_winrate", "hr_plcrate", ...]
for col in structural_cols:
    df[col] = df[col].fillna(0)

# Random missing -> train median
medians = df.loc[train_mask, remaining_cols].median()
df[remaining_cols] = df[remaining_cols].fillna(medians)

# Drop high-correlation features
drop_cols = ["chaksun2", "chaksun3", "chaksun4", "chaksun5", "buga2", "buga3", ...]
df = df.drop(columns=drop_cols)

# Time-based split 6:2:2
df["fold"] = assign_time_split(df, date_col="rcDate", ratios=(0.6, 0.2, 0.2))</pre>
</div>

<!-- ===== 2. 이상치 ===== -->
<h2>2. 이상치 처리 (v1~v4 vs v5~v8의 차이)</h2>

<div class="box">
<strong>이상치란?</strong> 다른 값들과 동떨어진 극단값.<br>
예) 평균 휴양일 42일인데 868일 쉰 말 = 이상치.<br><br>
<strong>탐지 방법:</strong> IQR (Q3 + 1.5xIQR 초과 또는 Q1 - 1.5xIQR 미만)
</div>

<table>
<tr><th>버전</th><th>이상치 처리</th><th>언제 쓰나</th></tr>
<tr><td>v1~v4</td><td><strong>이상치 포함</strong> (원본 그대로)</td><td>트리 모델(RF, XGBoost)은 이상치에 강건</td></tr>
<tr><td>v5~v8</td><td><strong>이상치 제거</strong> (IQR 밖 행 삭제)</td><td>로지스틱회귀, KNN 등 스케일 민감 모델</td></tr>
</table>

<p class="code-label">핵심 코드:</p>
<div class="code-block">
<pre># IQR outlier detection (fitted on train set only)
Q1 = train_df[col].quantile(0.25)
Q3 = train_df[col].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Remove rows outside IQR (only if outlier rate <= 30%)
for col, info in outlier_info.items():
    if info["pct"] <= 30:
        mask &= (df[col] >= info["lower"]) & (df[col] <= info["upper"])
df_clean = df[mask]</pre>
</div>

<div class="warn">
<strong>이상치 제거 기준:</strong><br>
- 이상치 비율 30% 초과 피처는 분포 자체가 넓은 것으로 간주, 제거 안 함<br>
- 배당률 관련 피처는 제외 (이미 model_features에서 빠져 있음)<br>
- 상세 결과는 <strong>results/eda/report.html</strong>의 3장 참고
</div>

<!-- ===== 3. 스케일링 ===== -->
<h2>3. 스케일링이 뭔가요?</h2>

<div class="box">
<p><strong>문제:</strong> "마체중"은 400~580 범위, "승률"은 0~1 범위. 로지스틱회귀 같은 모델은 숫자가 큰 컬럼에 더 영향을 받아 불공평해짐.</p>
<p><strong>해결:</strong> 모든 컬럼의 숫자 범위를 비슷하게 맞춰주는 게 "스케일링".</p>
</div>

<table>
<tr><th>방법</th><th>공식</th><th>특징</th></tr>
<tr><td><strong>없음</strong> (v1, v5)</td><td>원본 그대로</td><td>트리 모델은 스케일링 불필요</td></tr>
<tr><td><strong>StandardScaler</strong> (v2, v6)</td><td>(값 - 평균) / 표준편차</td><td>결과 대략 -3~+3. 가장 보편적.</td></tr>
<tr><td><strong>MinMaxScaler</strong> (v3, v7)</td><td>(값 - 최소) / (최대 - 최소)</td><td>결과 정확히 0~1. 거리 계산에 좋음.</td></tr>
<tr><td><strong>RobustScaler</strong> (v4, v8)</td><td>(값 - 중앙값) / IQR</td><td>극단값에 덜 흔들림.</td></tr>
</table>

<p class="code-label">핵심 코드:</p>
<div class="code-block">
<pre>from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()             # or MinMaxScaler() or RobustScaler()
scaler.fit(df.loc[train_mask, cols])  # fit on train only!
df[cols] = scaler.transform(df[cols]) # apply same transform to all</pre>
</div>

<!-- ===== 4. 전체 버전 표 ===== -->
<h2>4. 전체 버전 한눈에 보기</h2>

<table>
<tr><th>파일</th><th>이상치</th><th>스케일링</th><th>추천 모델</th></tr>
<tr><td>v1_base.csv</td><td>포함</td><td>없음</td><td>RF, XGBoost</td></tr>
<tr><td>v2_standard.csv</td><td>포함</td><td>Standard</td><td>로지스틱, SVM</td></tr>
<tr><td>v3_minmax.csv</td><td>포함</td><td>MinMax</td><td>KNN, K-means</td></tr>
<tr><td>v4_robust.csv</td><td>포함</td><td>Robust</td><td>이상치 강건</td></tr>
<tr style="background:#fff3e0;"><td>v5_base_no_outlier.csv</td><td><strong>제거</strong></td><td>없음</td><td>RF (정제)</td></tr>
<tr style="background:#fff3e0;"><td>v6_standard_no_outlier.csv</td><td><strong>제거</strong></td><td>Standard</td><td>로지스틱 (정제)</td></tr>
<tr style="background:#fff3e0;"><td>v7_minmax_no_outlier.csv</td><td><strong>제거</strong></td><td>MinMax</td><td>KNN (정제)</td></tr>
<tr style="background:#fff3e0;"><td>v8_robust_no_outlier.csv</td><td><strong>제거</strong></td><td>Robust</td><td>비교용</td></tr>
</table>

<div class="warn">
<strong>모르겠으면?</strong><br>
- 트리 모델(RF, XGBoost) -> <code>v1_base.csv</code><br>
- 로지스틱회귀 -> <code>v6_standard_no_outlier.csv</code><br>
- 여러 모델 비교 -> v1 vs v5로 이상치 영향 비교<br>
- KNN/K-means -> <code>v7_minmax_no_outlier.csv</code>
</div>

<!-- ===== 5. 사용법 ===== -->
<h2>5. 사용법</h2>

<p class="code-label">버전 파일을 직접 로드:</p>
<div class="code-block">
<pre>import pandas as pd

df = pd.read_csv("data/versions/v6_standard_no_outlier.csv", encoding="utf-8-sig")

train = df[df["fold"] == "train"]
valid = df[df["fold"] == "valid"]
test  = df[df["fold"] == "test"]

y_train = train["win"]
# X: drop ID/market/outcome columns (see src/1_전처리/config.py EXCLUDE_COLS)</pre>
</div>

<p class="code-label">또는 이미 분리된 파일 사용:</p>
<div class="code-block">
<pre>train = pd.read_csv("data/전처리_데이터셋/v6_standard_no_outlier/train.csv", encoding="utf-8-sig")
valid = pd.read_csv("data/전처리_데이터셋/v6_standard_no_outlier/valid.csv", encoding="utf-8-sig")
test  = pd.read_csv("data/전처리_데이터셋/v6_standard_no_outlier/test.csv",  encoding="utf-8-sig")</pre>
</div>

<!-- ===== 6. 주의사항 ===== -->
<h2>6. 주의사항</h2>

<div class="bad">
<strong>절대 하면 안 되는 것</strong><br>
- <code>fold</code> 컬럼을 무시하고 데이터를 섞는 것 -> 미래 정보 유출<br>
- valid/test로 스케일링 기준을 잡는 것 -> 이미 train 기준으로 처리됨<br>
- <code>win</code> 컬럼을 피처(X)로 사용 -> 이건 정답(y)임
</div>

<div class="footer">KHUDA 3조 · 전처리 버전 가이드 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    output_path = OUTPUT_DIR / "report.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Saved: {{output_path}}")


if __name__ == "__main__":
    main()
