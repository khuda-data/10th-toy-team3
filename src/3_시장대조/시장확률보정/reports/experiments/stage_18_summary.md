# 개발 18단계 결과 — 예측 출력 Schema

생성 시점: 2026-08-18 02:41 KST

## 구현된 출력 계약

```text
model_version
prediction_time
odds_snapshot_time
race_start_time
odds_source
race_id
entry_id
hrNo
hrName
winOdds_snapshot
q_market
p_premarket
p_final
market_delta
break_even_prob
expected_edge
pred_rank
action
rejection_reason
```

계산식:

```text
q_market = (1 / snapshot 배당) / 경주 내 역배당 합
market_delta = p_final - q_market
break_even_prob = 1 / winOdds_snapshot
expected_edge = p_final × winOdds_snapshot - 1
```

## 검증·거부 규칙

- `entry_id`는 전체 요청에서 유일해야 한다.
- 실제 입력 행 수는 경주별 선언 출전두수 `dusu`와 같아야 한다.
- 배당은 결측 없이 1보다 커야 한다.
- `odds_snapshot_time <= prediction_time < race_start_time`이어야 한다.
- 동일 경주의 시점 값은 모두 일치해야 한다.
- 유효 경주의 `q_market`, `p_premarket`, `p_final` 합은 각각 1이어야 한다.
- 경주 일부가 누락되거나 검증에 실패하면 해당 경주 전체를 `prediction_rejected` 처리한다.
- 결과·사후 열 `win`, `ord`, `fin_rank`, `fin_pct`, `resid`는 출력하지 않는다.

## 현재 정책

```text
model_version: m2_xgboost_sum_l005_t095_v1
model: M2 XGBoost
normalization: sum
market blend lambda: 0.05
temperature: 0.95
betting action: no_bet
```

역사적 fixture는 2경주 21두로 생성했으며 모든 유효 행의 행동은 `no_bet`이다. fixture의 배당은 실제 최종 배당이고 시점은 schema 검증을 위한 합성값이므로 운영 예측이나 라이브 백테스트로 사용할 수 없다.

## 라이브 운영 전 필수 과제

- 실제 예측 시점의 배당 snapshot과 timestamp 수집
- 권위 있는 경주 시작 시각과 전체 출전 명단 확보
- 입력 schema drift 및 `prediction_rejected` 모니터링

다음 개발 단계는 19단계 테스트·README 정리다.
