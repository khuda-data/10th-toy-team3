# 버전 비교 분석

전처리 버전 v1~v8을 직접 비교해, 이상치 제거 여부와 스케일링 방식이 모델 성능에 실제로 어떤 영향을 주는지 확인한 분석입니다.

## 코드 설명

| 파일 | 설명 | 산출물 | 작성자 |
|---|---|---|---|
| `run_analysis.py` | v1~v8 버전을 각각 읽어 RF/XGBoost/Logistic을 학습·평가하고, (1) 이상치 제거 효과(v1 vs v5, v2 vs v6 등) (2) 스케일링 방식별 성능을 비교해 HTML 보고서로 종합 | `results/analysis/` → `analysis_report.docx` | 박준석 (junseok) |
| `config.py` | 컬럼 분류·유틸리티 함수 등 파이프라인 공통 설정 모듈 (팀 공용 `src/pipeline/config.py`를 이 파이프라인에서 그대로 import해 사용) | — | 팀 공용 (원 작성자 미상) |

`analysis_report.docx`가 이 코드의 실행 결과입니다. `전처리 데이터셋/` v1~v8이 준비된 뒤에 실행합니다.
