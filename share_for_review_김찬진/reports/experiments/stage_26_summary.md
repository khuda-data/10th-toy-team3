# 개발 26단계 결과 — 시장 유지/교체 게이트

생성 시점: 2026-08-22T08:32:17.674925+00:00

## 구현

- 게이트 학습: Train OOF에서 시장/R2 1위가 다른 경주만 사용
- 목표 1: R2만 적중해 시장을 뒤집는 것이 유리한 경주
- 목표 0: 시장만 적중하거나 둘 다 실패한 경주
- Calibration 역할: gate threshold와 `lambda_switch` 선택
- 기본 행동: `lambda_race=0`, 시장 유지
- 선택 제약: 시장 대비 Race Log Loss와 Brier가 모두 악화되지 않아야 함

## 게이트 데이터

- Train OOF 불일치: 383경주, 유리한 교체 63경주
- Calibration 불일치: 329경주, 유리한 교체 55경주
- Calibration gate ROC-AUC `0.5986`, Average Precision `0.2228`

## 선택 결과

- 정책 상태: `candidate_pending_stage_27_and_future_holdout`
- threshold: `0.65`
- lambda_switch: `0.3`
- gate action: 33경주
- 실제 시장 Top-1 교체: 14경주
- 시장 Top-1 `247/641` → 후보 `249/641`
- Delta Log Loss `+0.001795`, Delta Brier `+0.000338`, Delta Top-1 `+0.31%p`

## 제약의 효과

정확도만 최대화하면 threshold `0.5`, lambda `0.3`에서 `252/641`로 시장보다 `+5`경주 높다. 그러나 Delta Log Loss `-0.000145`, Delta Brier `-0.000239`로 두 확률 제약을 통과하지 못해 탈락했다.

시장 유지 후보도 grid에 포함했다. 제약을 지키면서 적중을 늘리지 못하면 `no_change`가 정식 결과이며, 억지로 시장 순위를 뒤집지 않는다.
현재 선택값은 Calibration 결과를 보고 정한 예비 후보다. 시장 우위는 새 Future Holdout에서 검증하기 전까지 주장하지 않는다.

## 산출물

- `artifacts/models/r4_market_gate_logistic.joblib`: Train OOF 게이트 모델
- `data/analysis/stage_26_gate_grid.csv`: threshold/lambda 전체 후보
- `data/predictions/r4_gated_calibration.csv.gz`: 선택 정책 확률과 경주별 lambda
- `data/manifests/market_gate_policy.json`: 선택 정책과 해시
- `reports/experiments/stage_26_market_gate.json`: 학습·Calibration 상세 결과

다음 27단계에서는 시장, 기존 고정 혼합, R2, R4를 동일한 비열화 제약으로 최종 비교하고 새 미래 holdout 전에 후보를 동결한다.
