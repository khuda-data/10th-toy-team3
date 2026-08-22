# 개발 8~11단계 결과 요약

생성 시점: 2026-08-17 23:38 KST

## 실행 정책

- 입력: `eligible_primary=true`인 서울 경주의 완전한 경주 단위 데이터
- 피처: feature registry에서 `PRE_RACE`로 승인된 112개 열만 사용
- 전처리 적합: Train 또는 각 walk-forward Train 구간만 사용
- 모델 선택 검증: Train 내부의 날짜 정렬 4-fold walk-forward
- 비교 구간: Calibration 641경주
- Final Test: 미사용·봉인 유지

## 8단계 — Train 기준 전처리

- 수치형 99개: Train의 0.5%/99.5% 분위수로 clipping
- 결측치: Train 중앙값 대치 및 결측 indicator 추가
- 범주형 13개: Train 최빈값 대치, one-hot encoding, 빈도 10 미만 범주 통합
- 미지 범주: 추론 시 오류 없이 infrequent 범주로 처리
- M1: 수치형 StandardScaler 적용
- M2: 트리의 분기 구조를 보존하기 위해 scaling 생략
- 저장: 각 모델의 직렬화 artifact 안에 적합된 전처리기와 feature schema를 함께 저장

## 9~11단계 — Calibration 비교

| 모델 | Race Log Loss | M0 대비 | Race Brier | M0 대비 | Top-1 |
|---|---:|---:|---:|---:|---:|
| M0 시장 | 1.776036 | 기준 | 0.762636 | 기준 | 39.00% |
| M1 Logistic | 1.965904 | +0.189868 | 0.817458 | +0.054822 | 30.89% |
| M2 XGBoost | 1.976751 | +0.200715 | 0.821722 | +0.059086 | 29.64% |

낮을수록 좋은 두 주요 지표에서 M1과 M2 모두 M0 시장 기준을 넘지 못했다. M2는 Train Race Log Loss가 1.532825인데 Calibration은 1.976751이므로 과적합 신호도 크다. 이 결과는 실패한 실험까지 포함해 고정하며, Final Test를 확인해 모델을 선택하지 않는다.

## 생성 산출물

- `reports/experiments/m0_market_baseline.json`
- `reports/experiments/m1_logistic.json`
- `reports/experiments/m2_xgboost.json`
- `artifacts/models/m1_logistic.joblib`
- `artifacts/models/m2_xgboost.joblib`
- `data/predictions/m1_logistic_calibration.csv.gz`
- `data/predictions/m2_xgboost_calibration.csv.gz`

## 다음 판단

다음 개발 단계는 12단계인 독립 모델 확률의 경주 단위 정규화 방식 확정이다. 현재 비교에는 양의 raw 승리확률을 경주별 합으로 나누는 기본 정규화를 사용했다. 다음 단계에서는 이 방식과 logit softmax를 Train 내부 walk-forward로 비교하고, 선택 규칙을 고정한 뒤에만 Calibration 기반 시장 혼합으로 진행한다.
