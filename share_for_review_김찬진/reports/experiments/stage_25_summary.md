# 개발 25단계 결과 — R2 랭킹 점수 확률 변환

생성 시점: 2026-08-22T08:17:20.099477+00:00

## 선택 규칙

- 변환: 경주별 `softmax(ranking_score / T_rank)`
- 선택 데이터: Calibration 641경주만 사용
- 선택 지표: Race Log Loss 최소화, 동률이면 Brier와 T=1 근접성
- Temperature는 순위를 바꾸지 않으므로 Top-1은 24단계와 동일
- 선택 결과: `T_rank=0.65`

## Calibration 확률 성능

| 후보 | Race Log Loss ↓ | Race Brier ↓ | Top-1 | MRR |
|---|---:|---:|---:|---:|
| 시장 | 1.776036 | 0.762636 | 38.53% | 0.5778 |
| R2 softmax | 1.967320 | 0.819832 | 30.11% | 0.5058 |

시장 대비 R2 개선량은 Log Loss `-0.191283`, Brier `-0.057196`, Top-1 `-8.42%p`다. 양수는 R2 우위다.
기존 독립 M2 대비로는 Log Loss `+0.009432`, Brier `+0.001890`, Top-1 `+0.47%p`다.

## 해석

- R2 확률은 경주별 합이 1인 유효한 분포가 됐다.
- Temperature는 확률의 날카로움만 바꾸며 24단계의 말 순위를 바꾸지 않았다.
- Train OOF 수치는 나중 시점인 Calibration에서 선택한 T를 역적용한 참고값이므로 독립 선택 성능으로 사용하지 않는다.
- 시장 대비 Log Loss 또는 Brier가 음수이면 R2 단독 확률은 27단계 비열화 제약을 통과할 수 없다. 26단계에서 시장 유지가 기본인 적응형 결합을 검토한다.
- Calibration에서는 기존 M2보다 세 지표가 소폭 좋아졌지만, OOF 참고 구간에서는 R2 Log Loss와 Brier가 M2보다 나빠 개선이 안정적으로 재현되지 않았다.
- 기존 시장 혼합 모델의 Calibration Log Loss는 `1.773358`로 R2 단독보다 훨씬 낮다. 시장 앵커를 제거하면 확률 품질이 크게 악화된다.

## 참고 Train OOF

선택 T 역적용 R2 Log Loss `2.030491`, Brier `0.832781`. 기존 M2 대비 각각 `-0.025164`, `-0.006323`이며 공식 선택 근거가 아니다.

## 산출물

- `data/manifests/ranker_temperature_policy.json`: 선택 T와 사용 경계
- `data/predictions/r2_xgb_ranker_calibration_probability.csv.gz`: Calibration 확률
- `data/predictions/r2_xgb_ranker_train_oof_probability.csv.gz`: 선택 T 역적용 OOF 참고 확률
- `reports/experiments/stage_25_ranker_probability.json`: 전체 temperature grid와 비교 지표

다음 26단계에서는 시장 유지/교체 게이트와 경주별 lambda를 구현한다.
