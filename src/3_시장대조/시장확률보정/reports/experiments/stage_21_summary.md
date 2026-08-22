# 개발 21단계 결과 — Top-1 제약 연구 정책 동결

생성 시점: 2026-08-22

## 완료 항목

- `PROJECT_GUIDELINES.md`에 Top-1 제약 최적화 확장 지침 추가
- `data/manifests/top1_research_policy.json`에 데이터 사용 경계와 성공·승격 기준 고정
- 기존 Final Test를 설명 전용으로 잠그고 신규 후보 선택 사용 금지
- 새 Future Holdout을 2026-08-09 이후 첫 500개 적격 서울 경주로 사전 정의
- 고유 Top-1 동률 처리, 후보 필터 순서, paired bootstrap과 McNemar 검정 규칙 고정

## 고정된 연구 목표

```text
시장 대비 Race Log Loss와 Race Brier가 악화되지 않는 후보만 통과시킨다.
통과 후보 중 시장 대비 Top-1 개선이 가장 큰 후보를 선택한다.
공식 성공은 새 Future Holdout 500경주에서 단 한 번 검증한다.
```

개발 구간에서 Top-1이 높다는 사실만으로 성공을 선언하지 않는다. 새 holdout에서 Top-1 개선의 paired bootstrap 95% 신뢰구간 하한이 0보다 크고 exact McNemar 검정이 유의하며, Log Loss와 Brier도 시장보다 나쁘지 않아야 `Top-1 연구 성공`이다.

기존 공식 확률 기준까지 충족해야 새 챔피언으로 승격한다. 그렇지 않으면 `research_challenger`, `no_change` 또는 데이터가 덜 모인 경우 `pending_data`로 기록한다.

## 현재 상태

- 21단계 정책 동결: 완료
- 모델 학습·후보 선택: 아직 시작하지 않음
- 새 Future Holdout: `pending_data`
- 기존 챔피언: 변경 없음
- 베팅 정책: `no_bet` 유지

다음 단계는 22단계다. Train의 시간순 OOF 예측과 Calibration만 사용해 시장과 기존 모델의 1위 불일치 경주를 네 가지 정오 조합으로 분석한다.
