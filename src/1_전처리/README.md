# 전처리 코드

`data/전처리_데이터셋/`과 `reports/1_전처리/`를 만들어낸 EDA·전처리 파이프라인입니다. `01 → 07` 순서로 실행합니다.

## 코드 설명

| 파일 | 설명 | 산출물 | 작성자 |
|---|---|---|---|
| `01_missing_check.py` | 156개 컬럼 전체의 결측치 현황을 파악하고, 결측률 5% 이상 컬럼을 구조적 결측/랜덤 결측으로 분류 | — (분석용 CSV) | 박준석 (junseok) |
| `02_correlation_check.py` | 수치형 피처 상관행렬·히트맵을 그리고 `\|r\|>=0.8`인 고상관 쌍을 추출 (`--vif` 옵션으로 VIF 계산) | — (분석용 CSV) | 박준석 (junseok) |
| `03_eda_summary.py` | 01~02 단계 결과 CSV를 읽어 종합 요약(`summary.md`)을 자동 작성 | `summary.md` | 박준석 (junseok) |
| `04_eda_report.py` | 결측치 → 이상치 → 다중공선성 → 처리 방향을 종합한 EDA HTML 보고서 생성 | `results/eda/report.html` → `reports/1_전처리/eda_report.html` | 박준석 (junseok) |
| `05_make_versions.py` | 이상치 제거 여부(2종) × 스케일링 방식(4종)을 조합해 전처리 버전 8종의 CSV를 생성 | `data/versions/v1~v8_*.csv` | 박준석 (junseok) |
| `06_version_report.py` | 8개 버전 중 어떤 것을 언제 쓰면 되는지 안내하는 HTML 보고서 생성 | `data/versions/report.html` → `reports/1_전처리/versions_report.html` | 박준석 (junseok) |
| `07_split_data.py` | 8개 버전 CSV를 시간순 fold 컬럼 기준으로 train/valid/test로 물리 분리 | `data/전처리_데이터셋/<version>/{train,valid,test}.csv` | 박준석 (junseok) |
| `config.py` | 컬럼 분류·유틸리티 함수 등 파이프라인 공통 설정 모듈 (팀 공용 설정. 각 파이프라인이 독립 실행되도록 같은 파일을 폴더마다 복사해 뒀습니다 — 7곳 모두 내용이 동일합니다) | — | 팀 공용 (원 작성자 미상) |

## 실행 순서

`01 → 02 → 03 → 04 → 05 → 06 → 07` 순서로 실행합니다. 01~03은 EDA 분석, 04는 그 결과를 보고서로 정리, 05~07은 분석 결과를 반영해 실제 전처리 데이터셋을 만드는 단계입니다.

01~04는 원천 데이터 `data/raw/race_entries.csv.gz`를 입력으로 씁니다. 경로는 `config.py`의 `RAW_ENTRIES` 상수에
들어 있어 어느 위치에서 실행하든 같은 파일을 찾습니다. 05~07은 앞 단계가 만든 `data/processed/`·`data/versions/`를 읽습니다.
