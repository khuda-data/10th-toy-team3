# 개발 24단계 결과 — R2 Pairwise Ranker

생성 시점: 2026-08-22T08:09:15.383263+00:00

## 학습 계약

- 목적함수: `rank:pairwise`
- 평가 관점: 경주 내 우승마를 다른 출전마보다 위에 배치
- 피처: 시장 정보가 없는 112개 PRE_RACE 피처
- Train: 4-fold expanding walk-forward OOF
- Calibration: 전체 Train 학습 후 독립 평가
- 기존 Final Test: 사용하지 않음
- R2 출력은 아직 확률이 아니며 Race Log Loss·Brier는 25단계 전까지 계산하지 않음

## 순위 성능

| 구간 | 후보 | Top-1 | 적중수 | Hit@3 | MRR | 우승마 평균순위 |
|---|---|---:|---:|---:|---:|---:|
| Train OOF | R0_market | 36.88% | 284 | 67.92% | 0.5626 | 2.987 |
| Train OOF | R1_existing_m2 | 27.92% | 215 | 60.13% | 0.4915 | 3.482 |
| Train OOF | R2_pairwise_ranker | 28.05% | 216 | 58.44% | 0.4885 | 3.571 |
| Calibration | R0_market | 38.53% | 247 | 70.51% | 0.5778 | 2.891 |
| Calibration | R1_existing_m2 | 29.64% | 190 | 62.25% | 0.5029 | 3.393 |
| Calibration | R2_pairwise_ranker | 30.11% | 193 | 63.03% | 0.5058 | 3.367 |

## 시장 대비 결과

- Train OOF: R2 Top-1 적중 `-68`경주, `-8.83%p` 차이
- Calibration: R2 Top-1 적중 `-54`경주, `-8.42%p` 차이

## 안정성 해석

- R2의 기존 M2 대비 fold별 Top-1 적중 차이는 `-3, -2, +4, +2`경주였다.
- Train OOF 합계에서는 M2보다 `+1`경주, Calibration에서는 `+3`경주였지만 네 fold에서 방향이 일관되지 않았다.
- 시장은 네 OOF fold 모두에서 R2보다 높았다. R2 단독 순위를 시장 대체 모델로 사용할 근거는 없다.
- 다만 Calibration의 Hit@3·MRR·우승마 평균순위도 기존 M2보다 소폭 개선되어 25단계 확률 변환 후보로는 유지한다.

24단계 결과는 랭킹 모델 자체의 진단이다. 후보 승격이나 시장 우위 선언이 아니며, 25단계에서 경주별 확률로 변환한 뒤 Log Loss와 Brier 비열화 제약을 적용해야 한다.

## 산출물

- `artifacts/models/r2_xgb_ranker.joblib`: 전체 Train 학습 전처리기와 ranker
- `data/predictions/r2_xgb_ranker_train_oof.csv.gz`: 시간순 OOF 랭킹 점수
- `data/predictions/r2_xgb_ranker_calibration.csv.gz`: Calibration 랭킹 점수
- `reports/experiments/stage_24_ranker.json`: fold별·통합 비교와 파일 해시

다음 25단계에서는 랭킹 점수를 경주별 softmax 확률로 바꾸고 Calibration Race Log Loss로 temperature를 선택한다.
