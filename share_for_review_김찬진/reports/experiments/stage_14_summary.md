# 개발 14단계 결과 — Temperature Scaling

생성 시점: 2026-08-18 01:18 KST

## 선택 규칙

```text
p_i(T) = softmax(log(p_i) / T)
```

- 입력: 13단계에서 고정한 `lambda=0.05` 시장 혼합 확률
- 선택 데이터: 고정 Calibration 641경주
- 온도 후보: 0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.50
- 선택 지표: Race Log Loss, 동률이면 Race Brier, 다시 동률이면 `T=1`에 가까운 값
- `T=1`: 보정하지 않은 기존 혼합 확률
- Final Test: 미사용·봉인 유지

## 선택 결과

| 후보 | lambda | 선택 T | Race Log Loss ↓ | Race Brier ↓ | Top-1 |
|---|---:|---:|---:|---:|---:|
| M0 시장 | 0.00 | 1.00 | 1.776036 | 0.762636 | **39.00%** |
| M1 최종 | 0.05 | 0.95 | 1.773767 | **0.761996** | 38.38% |
| M2 최종 | 0.05 | 0.95 | **1.773358** | 0.762075 | 38.22% |

두 모델 모두 `T=0.95`가 선택됐다. 이는 기존 혼합 분포를 약간 더 날카롭게 만드는 보정이다. Temperature Scaling은 경주 내 순서를 바꾸지 않으므로 Top-1과 MRR은 보정 전후 동일하다.

## 현재 최종 후보

```text
model: M2_xgboost
normalization: sum
market blend lambda: 0.05
temperature: 0.95
Calibration Race Log Loss: 1.773358
```

시장 단독 대비 Calibration Log Loss 개선은 약 `0.002678`, Brier 개선은 약 `0.000561`이다. 다만 lambda와 temperature를 동일한 Calibration fold에서 순차적으로 선택했으므로 이 개선은 낙관적으로 추정됐을 수 있다. 따라서 아직 시장 우위를 확정하지 않으며 Final Test와 경주 단위 bootstrap 전까지는 후보 상태로 유지한다.

다음 개발 단계는 15단계 Final Test 1회 평가다. 실행 전 현재 정책 파일과 모델 artifact의 체크섬을 동결하고, 평가 후에는 모델·lambda·temperature를 변경하지 않아야 한다.
