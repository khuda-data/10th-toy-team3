# 박준석 — 작업 총정리

박준석(GitHub: `junseok`)이 지금까지 진행한 데이터 수집·전처리·모델링·분석 전체를 담은 폴더입니다. 프로젝트 진행 순서대로 정리했습니다. 각 하위 폴더에는 파일별 설명을 담은 `README.md`가 따로 있습니다.

## 진행 순서와 폴더 지도

| 순서 | 폴더/파일 | 무엇을 했나 | 코드 | 결과물 |
|---|---|---|---|---|
| 1 | `데이터 수집·전처리 코드/` | KRA 공공데이터 API로 경주 데이터 수집, `final.csv`를 용도별 파일로 분리 | `kra_client.py`, `collect_rc_race.py`, `preprocess_final.py` | — (데이터 산출물, 용량 문제로 코드만 업로드) |
| 2 | `전처리 코드/` + `전처리 보고서/` | 결측·이상치·다중공선성 분석, 전처리 버전 8종(v1~v8) 생성, train/valid/test 분할 | `01_missing_check.py` ~ `07_split_data.py` | `eda_report.docx`, `versions_report.docx`, `summary.docx` (→ `전처리 데이터셋/`) |
| 3 | `1차 모델링 코드/` | 로지스틱회귀·랜덤포레스트·XGBoost로 1착 예측, 배당률 시장확률과의 괴리 분석, 군집화, Favorite-Longshot Bias 검증 | `01_train_model.py` ~ `05_report.py` | `report.docx` (1차 모델링 결과) |
| 4 | `버전 비교 분석/` | v1~v8 버전을 직접 비교해 이상치 제거·스케일링 방식의 실제 효과 검증 | `run_analysis.py` | `analysis_report.docx` |
| 5 | `v1~v3 실험 코드/` | 방향 1(확률 보정)·방향 2(시장 결합)·방향 3(이변 예측) 세 갈래 실험 | `v1_calibration/`, `v2_stacked/`, `v3_upset/`, `06_reports_v123.py` | `v1_report.docx`, `v2_report.docx`, `v3_report.docx` |
| 6 | `이변예측모델_배당률포함/` | 배당률에서 뽑은 시장확률(q)을 피처로 추가한 이변 예측 모델 — A/B/C 피처셋 비교, threshold 튜닝, 배당 구간별 검증, ROI 시뮬레이션 | `01_odds_feature_check.py` ~ `10_make_docx_report.py` | `이변_예측모델_배당률포함_보고서.docx`, `preprocessing_report.docx` |
| — | `경주마_모델_1차 결과보고서.docx` | 1차 모델링 결과의 별도 정리본 (수작업 작성, 대응 소스 코드 없음) | — | — |

## 공통 참고사항

- `config.py`는 여러 폴더에 반복해서 들어있습니다. 팀 공용 파이프라인 설정 모듈이라 폴더마다 실행에 필요해서 각각 복사해뒀습니다 (원본은 하나입니다).
- 모든 코드는 프로젝트 루트의 `final.csv`(또는 `data/processed/`의 분리된 파일)를 입력으로 사용합니다. 학습된 모델(`.pkl`)이나 원본 데이터는 용량 문제로 이 저장소에 올리지 않았습니다 — 코드를 실행하면 재생성됩니다.
- HTML로 생성되는 보고서는 모두 docx로 변환해서 올렸습니다. 표·그래프는 정상적으로 옮겨졌는지 확인했습니다.
