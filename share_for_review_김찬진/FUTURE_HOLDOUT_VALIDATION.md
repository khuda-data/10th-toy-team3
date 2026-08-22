# 새로운 최종 성능 및 시장 우위 공식 검증 절차

이 문서는 이미 공개된 기존 Final Test를 개발 자료로만 취급하고, 새로운 미래 경주에서 모델의 최종 성능과 시장 대비 우위를 검증하기 위한 사전등록 절차다. 후보 선택이 끝난 뒤 28단계부터 적용한다.

## 1. 검증 목적과 데이터 경계

- 공식 검증 대상은 2026-08-09 이후 시간순으로 수집되는 첫 500개 적격 서울 경주다.
- 기존 Final Test(2025-12-28~2026-08-09)는 `Legacy Test`로 명칭을 바꾸며 공식 검증에 사용하지 않는다.
- 적격 경주는 완전한 출전 목록, 정확히 한 마리의 우승마, 유효한 단승 배당, 예측 시점 이전 피처를 갖춘 정상 경주다.
- 취소·무효·공동우승·불완전 수집 경주는 결과를 보기 전에 경주 전체를 제외하고 사유를 기록한다.
- 후보 모델, 피처, 전처리, temperature, gate threshold, lambda, seed, 동률 처리 및 평가 코드는 첫 공식 예측 전에 동결한다.

## 2. 사전 예측 저장 계약

각 경주는 결과가 공개되기 전에 다음 정보를 저장한다.

- `race_id`, `entry_id`, `rcDate`
- 입력 피처 스냅숏 해시와 시장배당 스냅숏
- `odds_snapshot_time`, `prediction_time`, `race_start_time`
- `q_market`, 독립 랭커 점수와 확률, gate score와 action, 경주별 lambda, `p_final`
- 후보 정책 버전, 모델 파일 해시, 실행 코드 버전

시간 조건은 `odds_snapshot_time <= prediction_time < race_start_time`이어야 한다. 경주별 `q_market`과 `p_final`의 합은 각각 1이어야 하며, 입력 행 누락이나 중복이 있으면 해당 경주의 예측을 전부 거부한다.

## 3. 고유 Top-1 및 지표 계약

모델 Top-1은 `p_final` 내림차순, `q_market` 내림차순, `entry_id` 오름차순으로 정확히 한 마리를 선택한다. 시장 Top-1은 `q_market` 내림차순, `entry_id` 오름차순으로 선택한다.

다음 개선량은 양수가 모델 우위다.

```text
Delta Top-1   = Model Top-1 - Market Top-1
Delta LogLoss = Market Race Log Loss - Model Race Log Loss
Delta Brier   = Market Race Brier - Model Race Brier
```

## 4. 봉인과 중간 점검

- 500경주가 모일 때까지 정답을 이용한 누적 성능표를 모델 개발자에게 공개하지 않는다.
- 100경주 단위 점검은 입력 누락, 확률합, timestamp, 파일 해시 같은 운영 무결성만 확인한다.
- 중간 Top-1, Log Loss, Brier 또는 손익을 보고 모델·임계값·lambda를 바꾸거나 수집을 조기 종료하지 않는다.
- 불가피하게 정책을 변경하면 기존 회차를 종료하고 변경된 모델로 새로운 공식 검증 회차를 처음부터 시작한다.
- 예측 파일은 append-only로 보존하며 수정이 필요하면 원본을 남기고 수정 사유와 새 해시를 기록한다.

## 5. 500경주 단일 평가

500개 적격 경주를 채운 뒤 정답을 한 번만 결합하고 다음을 계산한다.

1. 시장과 후보의 Race Log Loss, Race Brier, Top-1 accuracy
2. 세 지표의 시장 대비 paired 경주별 차이
3. 경주 단위 paired bootstrap 10,000회, seed 42의 95% 신뢰구간
4. 시장과 모델의 Top-1 정오표에 대한 양측 exact McNemar 검정
5. 월·거리·등급·주로 상태별 보조 안정성 분석
6. 모든 제외 경주와 데이터 품질 위반 내역

Future Holdout을 연 뒤 bootstrap 반복 수, seed, 유의수준, 세그먼트 또는 성공 기준을 변경하지 않는다.

## 6. 판정 기준

Top-1 연구 성공은 다음을 모두 만족해야 한다.

- `Delta Top-1 > 0`
- Top-1 paired bootstrap 95% 신뢰구간 하한이 0보다 큼
- 양측 exact McNemar 검정 `p < 0.05`
- `Delta LogLoss >= 0`, `Delta Brier >= 0`
- 확률합·고유 Top-1·시간순·누수 검사를 모두 통과하고 주요 구간에 치명적 붕괴가 없음

새 챔피언 승격은 위 조건에 더해 `Delta LogLoss > 0`, Log Loss paired bootstrap 95% 신뢰구간 하한 0 이상, `Delta Brier > 0`, 기간별 개선 재현과 심한 calibration 왜곡 없음이 필요하다. Top-1 조건만 통과하면 `research_challenger`, 전체 조건을 통과하지 못하면 `no_change`로 기록한다. 경제적 우위는 별도 사전등록 백테스트 없이는 주장하지 않는다.

## 7. 기존 Final Test의 허용 범위

기존 Final Test는 오류 분석, 민감도 검사 또는 동결 후 운영 모델 재학습에 활용할 수 있다. 단, 이 데이터를 이용해 얻은 수치는 `retrospective exploratory evidence`로 표시하며 새로운 최종 성능이나 시장 우위의 공식 증거로 사용하지 않는다. 동결 후 재학습하는 경우에도 각 과거 경주의 gate 학습 입력은 해당 경주 이전 데이터만 사용한 walk-forward/OOF 예측이어야 한다.

## 8. 완료 산출물

- 동결 후보 정책과 모든 구성 파일의 SHA-256
- 결과 공개 전 생성된 500경주의 예측 원장
- 적격·제외 경주 manifest와 제외 사유
- 단일 공식 평가 보고서와 bootstrap 원표본
- McNemar 정오표와 검정 결과
- `research_success`, `research_challenger`, `champion_promoted`, `no_change` 중 최종 상태

본 절차는 `PROJECT_GUIDELINES.md` 29절과 `data/manifests/top1_research_policy.json`보다 기준을 완화하지 않는다. 충돌이 있으면 더 엄격한 규칙을 적용한다.
