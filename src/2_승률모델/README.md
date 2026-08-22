# 1차 모델링 코드

`박준석/report.docx`(1차 모델링 결과)를 만든 코드입니다. 로지스틱회귀/랜덤포레스트/XGBoost로 1착 여부를 예측하고, 배당률의 시장확률과 비교합니다.

## 코드 설명

| 파일 | 설명 | 산출물 | 작성자 |
|---|---|---|---|
| `01_train_model.py` | 데이터 로드·Train/Valid/Test 분할 → 로지스틱회귀/랜덤포레스트/XGBoost 학습·평가 → valid set 기준 threshold 튜닝, test set 최종 평가 | `results/models/` | 박준석 (junseok) |
| `02_market_gap.py` | test set 예측확률과 배당률의 시장 암묵적확률 사이의 괴리(gap)를 계산하고, feature importance를 괴리와 함께 재해석 | `results/market_gap.csv` 등 | 박준석 (junseok) |
| `03_clustering.py` | 괴리 절대값 상위 20% 서브셋을 K-means로 군집화 (실루엣 계수로 최적 k 탐색), 군집별 피처 특성 분석 | `results/cluster_profiles.csv` 등 | 박준석 (junseok) |
| `04_validation.py` | 괴리 구간별 모델확률·시장확률·실제 승률을 비교하고, Favorite-Longshot Bias(인기마 과소평가·비인기마 과대평가 경향)를 분석 | `results/flb_*.csv/png` | 박준석 (junseok) |
| `05_model_report.py` | 위 결과를 하나의 HTML 보고서로 종합 | `results/report.html` → `박준석/report.docx` | 박준석 (junseok) |
| `config.py` | 컬럼 분류·유틸리티 함수 등 파이프라인 공통 설정 모듈 (팀 공용 `src/pipeline/config.py`를 이 파이프라인에서 그대로 import해 사용) | — | 팀 공용 (원 작성자 미상) |

## 실행 순서

`01 → 02 → 03 → 04 → 05` 순서로 실행합니다. `01_train_model.py`가 만든 `results/models/`는 `v1~v3 실험 코드/`의 세 방향(확률 보정·시장 결합·이변 예측) 실험이 그대로 이어받아 사용합니다.
