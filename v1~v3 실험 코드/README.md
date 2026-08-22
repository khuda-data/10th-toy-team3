# v1~v3 실험 코드

루트의 `v1_report.html` / `v2_report.html` / `v3_report.html`(방향 1/2/3 실험)을 만든 코드입니다.

**전제 조건** — 세 방향 모두 `01_train_model.py`(1차 모델링, 루트 `report.html`)로 학습된 RandomForest 모델(`results/models/`)을 불러와서 시작합니다. 그 코드는 이 폴더에는 포함하지 않았습니다.

## 코드 설명

| 파일 | 설명 | 산출물 | 작성자 |
|---|---|---|---|
| `v1_calibration/run.py` | 방향 1 — 기존 RandomForest 모델의 예측 확률을 Platt Scaling(시그모이드)·Isotonic Regression으로 보정해, 실제 승률에 더 가까워지는지 검증 | `results/v1_calibration/` → `v1_report.html` (확률 보정 실험) | 박준석 (junseok) |
| `v2_stacked/run.py` | 방향 2 — 1단계 모델의 예측확률과 배당률에서 뽑은 시장확률(q)을 결합한 2단계 모델을 만들어, 시장보다 나은 확률 추정이 가능한지 검증 | `results/v2_stacked/` → `v2_report.html` (시장 결합 실험) | 박준석 (junseok) |
| `v3_upset/run.py` | 방향 3 — 목표를 "1착 맞히기"에서 "시장이 틀리는 경우 찾기"로 바꿔, 비인기마(`pop_pct >= 0.5`)의 1착(`upset_B`)을 예측하는 모델 검증 | `results/v3_upset/` → `v3_report.html` (이변 예측 실험) | 박준석 (junseok) |
| `06_reports_v123.py` | v1~v3 결과를 각각 보기 좋은 HTML로 재생성 (그래프 영문화, 피처명 한글 설명 변환, 쉬운 말 설명 추가) | `v1_report.html`, `v2_report.html`, `v3_report.html` | 박준석 (junseok) |
| `config.py` | 컬럼 분류·유틸리티 함수 등 파이프라인 공통 설정 모듈 (팀 공용 `src/pipeline/config.py`를 이 파이프라인에서 그대로 import해 사용) | — | 팀 공용 (원 작성자 미상) |

## 실행 순서

`01_train_model.py`로 기본 모델을 먼저 학습해야 합니다. 그 뒤 `v1_calibration/run.py`, `v2_stacked/run.py`, `v3_upset/run.py`는 서로 독립적이라 순서 없이 실행 가능하며, 마지막에 `06_reports_v123.py`를 실행하면 세 결과를 정리한 HTML이 나옵니다.
