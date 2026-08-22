# 개발 27단계 결과 — Top-1 후보 비교 및 사전 동결

생성 시점: 2026-08-22T08:46:01.408499+00:00

## 데이터 정책

- 후보 선택: Calibration 6,582행·641경주만 사용
- 기존 Final Test: 로드·재평가·선택에 사용하지 않음
- 모든 후보: 동일 entry_id, 고유 Top-1 동률 처리, 경주별 확률합 1 검증

## 동일 기준 비교

| 후보 | Log Loss | Brier | Top-1 | 시장 대비 적중 | 실제 교체 | 제약 통과 |
|---|---:|---:|---:|---:|---:|---:|
| R0_market | 1.776036 | 0.762636 | 247/641 (38.53%) | +0 | 0 | 통과 |
| R1_existing_m2_standalone | 1.976751 | 0.821722 | 190/641 (29.64%) | -57 | 333 | 탈락 |
| R2_ranker_probability | 1.967320 | 0.819832 | 193/641 (30.11%) | -54 | 329 | 탈락 |
| R3_fixed_m2_market_blend | 1.773358 | 0.762075 | 245/641 (38.22%) | -2 | 13 | 통과 |
| R4_gated_adaptive_blend | 1.774241 | 0.762298 | 249/641 (38.85%) | +2 | 14 | 통과 |

## 동결 결과

- challenger: `r4_gate_ranker_t065_gate065_l030_v1`
- 선택 후보: `R4_gated_adaptive_blend`
- gate threshold: `0.65`
- lambda_switch: `0.3`
- rank temperature: `0.65`
- 시장 대비: Log Loss `+0.001795`, Brier `+0.000338`, Top-1 `+2`경주
- 상태: `frozen_pending_future_holdout`

R4는 확률 비열화 제약을 통과한 후보 중 Top-1 적중이 가장 높아 선택됐다. 이 결과는 Calibration 선택 결과이며 공식 시장 우위가 아니다. 기존 챔피언과 `no_bet` 정책은 유지한다.

새로운 공식 검증은 `FUTURE_HOLDOUT_VALIDATION.md`에 사전등록된 절차와 2026-08-09 이후 첫 500개 적격 경주를 사용한다.
