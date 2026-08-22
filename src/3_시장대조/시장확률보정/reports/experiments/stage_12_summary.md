# 개발 12단계 결과 — 경주 단위 확률 정규화

생성 시점: 2026-08-17 23:53 KST

## 선택 절차

- 후보 1 `sum`: 개별 승리확률을 같은 경주의 확률 합으로 나눔
- 후보 2 `logit_softmax`: 개별 승리확률의 logit에 경주 내 softmax 적용
- 선택 데이터: Train 내부 4-fold walk-forward OOF 770경주
- 선택 지표: Race Log Loss, 동률이면 Race Brier, 다시 동률이면 단순한 `sum`
- Calibration과 Final Test는 선택에 사용하지 않음

## Train OOF 결과

| 모델 | 정규화 | Race Log Loss ↓ | Race Brier ↓ | 선택 |
|---|---|---:|---:|---|
| M1 Logistic | sum | **2.007042** | **0.828100** | 선택 |
| M1 Logistic | logit softmax | 2.015960 | 0.833320 |  |
| M2 XGBoost | sum | **2.005327** | **0.826459** | 선택 |
| M2 XGBoost | logit softmax | 2.020754 | 0.833731 |  |

두 모델 모두 `sum` 정규화가 Log Loss와 Brier에서 일관되게 우수해 최종 정책으로 고정했다.

## 선택 후 Calibration 진단

| 모델 | sum Log Loss | logit softmax Log Loss | sum 우위 |
|---|---:|---:|---:|
| M1 Logistic | 1.965904 | 1.973049 | 0.007145 |
| M2 XGBoost | 1.976751 | 1.991527 | 0.014776 |

Calibration 결과도 Train OOF의 선택 방향과 일치하지만, 이 값은 선택 과정에는 사용하지 않았다. Final Test는 계속 봉인한다.

## 고정 정책

```text
M1_logistic: sum
M2_xgboost: sum
```

다음 개발 단계는 13단계인 Calibration 기반 시장 확률과 모델 확률의 기하 혼합 `lambda` 선택이다. `lambda=0`을 반드시 후보로 포함하며, 시장만 사용하는 결과도 유효한 결론으로 인정한다.
