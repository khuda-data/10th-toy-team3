# 개발 13단계 결과 — 시장 확률 기하 혼합

생성 시점: 2026-08-18 00:47 KST

## 선택 규칙

```text
p_i(lambda) ∝ q_market_i^(1-lambda) × p_model_i^lambda
```

- 선택 데이터: 고정 Calibration 641경주
- lambda 후보: 0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00
- 선택 지표: Race Log Loss, 동률이면 Race Brier, 다시 동률이면 더 작은 lambda
- `lambda=0`: 시장만 사용
- `lambda=1`: 비시장 모델만 사용
- Final Test: 미사용·봉인 유지

## 모델별 선택 결과

| 후보 | 선택 lambda | Race Log Loss ↓ | Race Brier ↓ | Top-1 |
|---|---:|---:|---:|---:|
| M0 시장 | 0.00 | 1.776036 | 0.762636 | **39.00%** |
| M1 Logistic 혼합 | 0.05 | 1.775646 | **0.762462** | 38.38% |
| M2 XGBoost 혼합 | 0.05 | **1.775286** | 0.762541 | 38.22% |

Race Log Loss 기준 최종 후보는 `M2 XGBoost + 시장`, `lambda=0.05`다. 시장 단독 대비 Log Loss 개선은 약 `0.000750`, Brier 개선은 약 `0.000095`에 불과하고 Top-1은 0.78%p 낮다. 따라서 현재 결과는 시장보다 우수하다고 확정할 근거가 아니라, 후속 보정과 bootstrap 검증 대상으로 취급한다.

## 고정 정책

```text
M1_logistic lambda: 0.05
M2_xgboost lambda: 0.05
현재 deployment candidate: M2_xgboost, lambda=0.05
```

다음 개발 단계는 14단계 temperature scaling이다. Calibration에서 이미 lambda를 선택했으므로 과도한 추가 탐색을 피하고, 사전에 정한 온도 격자와 시장 단독을 포함한 보수적 선택 규칙을 사용해야 한다.
