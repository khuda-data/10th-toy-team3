# 파트 1 — 독립 모델의 확률 품질

배당률·발매금액 등 당일 시장 정보를 입력하지 않은 Logistic Regression과 XGBoost가 각 말의 승리 확률을 얼마나 잘 표현하는지 검증한 최종 발표용 패키지입니다.

## 발표에서 사용하는 흐름

1. 정적 피처 레지스트리에서 `PRE_RACE` 역할만 선택합니다.
2. Logistic Regression과 XGBoost를 시간순 Train 구간으로 학습합니다.
3. 말별 이진 예측을 경주별 합이 1인 확률로 정규화합니다.
4. Train walk-forward OOF에서 정규화 방법을 비교하고 두 모델 모두 sum normalization을 선택합니다.
5. Final Test 6,639두·635경주에서 독립 모델과 시장 기준선의 Race Log Loss·Race Brier를 비교합니다.

## 독립 모델 Final Test 결과

| 후보 | Race Log Loss ↓ | Race Brier ↓ | Top-1 |
|---|---:|---:|---:|
| 시장 | **1.817348** | **0.777920** | **37.80%** |
| Logistic 독립 모델 | 1.965989 | 0.820059 | 31.02% |
| XGBoost 독립 모델 | 1.948620 | 0.812674 | 32.44% |

두 독립 모델 모두 시장보다 손실이 높았습니다. 시장 혼합·온도 보정 뒤의 수치는 이 표에 포함하지 않습니다.

## 남긴 코드

- `src/data/`: 원본 로드, 서울 데이터 구성, 시간순 분할
- `src/features/`: 시장·사후 정보 차단 레지스트리와 Train 전용 전처리
- `src/evaluation/race_metrics.py`: 경주 확률 정규화와 Log Loss·Brier
- `src/models/train_m1_logistic.py`: Logistic 독립 모델
- `src/models/train_m2_xgboost.py`: XGBoost 독립 모델
- `src/models/select_normalization.py`: sum/logit-softmax 비교
- `src/models/market_baseline.py`: 비교 기준선
- `발표근거/`: 독립 모델 Test 수치와 원본 산출물 위치

## 기초 데이터와 테스트 재생성

저장소 중복을 줄이기 위해 서울 중간 데이터·분할표·피처 레지스트리는 Git에 넣지 않습니다. 저장소 루트의 `data/race_entries.csv.gz`에서 아래 순서로 다시 만듭니다.

```bash
python -m src.data.build_seoul_interim
python -m src.data.build_splits
python -m src.features.build_registry
python -m unittest discover -s tests -v
```

위 절차는 원본을 수정하지 않고 `2_확률품질/data/` 아래의 제외된 파생파일만 생성합니다.

시장 혼합, temperature scaling, 랭커, 게이트, 베팅 백테스트 코드는 최종 발표 범위가 아니므로 `master`에서 제거했고 `main`과 Git 이력에 보존됩니다.
