# 개발 23단계 결과 — 경주 그룹 랭킹 데이터 계약

생성 시점: 2026-08-22T07:54:20.454262+00:00

## 완료 항목

- `race_id`별 출전마를 연속 배치하는 `RankingDataset` 구현
- 우승마 relevance 1, 나머지 0인 이진 랭킹 타깃 고정
- XGBoost `group` 인자용 출전두수 배열과 행 구간 manifest 생성
- 112개 PRE_RACE 피처만 허용하고 MARKET·POST_RACE·TARGET 입력 차단
- Train 내부 4-fold가 날짜순·경주 비중첩인지 사전 검증
- 기존 Final Test와 비정상 경주를 랭킹 데이터에서 제외

## 데이터 규모

| Fold | 행 | 경주 그룹 | 기간 | 최소/중앙/최대 출전두수 | relevance 합 |
|---|---:|---:|---|---:|---:|
| Train | 19,617 | 1,891 | 20230805~20250511 | 7/11/16 | 1,891 |
| Calibration | 6,582 | 641 | 20250517~20251227 | 6/11/16 | 641 |

각 경주에는 relevance 1이 정확히 하나 있으며, 모든 group size의 합은 해당 fold 행 수와 일치한다. 전처리 통계는 저장하지 않았고 24단계에서 각 walk-forward 학습 부분에만 적합한다.

## 산출물

- `data/interim/ranking_entry_manifest.csv.gz`: 랭킹 행 순서·그룹 위치·relevance
- `data/interim/ranking_group_manifest.csv`: 경주별 행 시작/종료·출전두수·우승마 위치
- `data/manifests/ranking_dataset_manifest.json`: 피처·fold·group·해시 계약
- `src/data/ranking_data.py`: 빌더와 fail-closed 검증 함수

다음 24단계에서는 이 계약을 사용해 R2 pairwise ranker를 시간순 4-fold로 학습하고 시장·기존 M2와 비교한다.
