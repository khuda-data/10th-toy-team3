# 이변 예측 모델 — 배당률 포함

비인기마의 1착(이변)을 예측할 때, 배당률에서 뽑아낸 시장 확률(q)을 피처로 더하면 기존 피처만 쓸 때보다 얼마나 개선되는지 검증한 실험입니다.

- 최종 보고서 — [`reports/4_이변모델/04_배당률포함_모델.docx`](../../reports/4_이변모델/04_배당률포함_모델.docx)
- 전처리 보고서 — [`reports/4_이변모델/03_배당률포함_전처리.docx`](../../reports/4_이변모델/03_배당률포함_전처리.docx)
- 이 폴더의 `01 → 10`이 위 보고서를 만들어낸 실행 파이프라인입니다

## 코드 설명

| 파일 | 설명 | 작성자 |
|---|---|---|
| `01_odds_feature_check.py` | 배당률에서 파생한 14개 컬럼 간 상관관계를 확인하고, 대표 피처로 정규화 암묵적 확률 `q`를 선정 | 박준석 (junseok) |
| `02_data_prep.py` | 서울 데이터만 필터링하고 비인기마(`pop_pct >= 0.5`)만 남긴 뒤 이변(비인기마 1착) 타겟을 생성, 시간순 6:2:2 분할과 피처셋 A/B/C 정의 | 박준석 (junseok) |
| `03_train_compare.py` | A(배당률 q 단독) / B(기존 피처만) / C(q + 기존 피처 결합) 세 피처셋을 Logistic Regression·Random Forest로 학습해 test set에서 비교 | 박준석 (junseok) |
| `04_threshold_tuning.py` | 최종 선정된 C 모델의 분류 임계값을 valid set의 F1(Macro) 기준으로 튜닝하고 test set에서 튜닝 전후 성능을 비교 | 박준석 (junseok) |
| `05_longshot_segment_eval.py` | 비인기마를 배당 구간별로 나눠 A 모델과 C 모델의 성능 차이를 비교 — 고배당 구간에서 배당률 피처 결합의 개선폭이 커지는지 검증 | 박준석 (junseok) |
| `06_feature_importance.py` | C 모델(Random Forest)의 Feature Importance 상위 20개를 시각화하고 `q`의 중요도 순위를 B 모델과 비교 | 박준석 (junseok) |
| `07_upset_report.py` | 01~06 단계의 결과를 하나의 HTML 보고서(`report.html`)로 종합 | 박준석 (junseok) |
| `08_full_pipeline.py` | 전처리(필터링·결측치 처리·다중공선성 제거·인코딩·스케일링·시간순 분할)부터 A/B/C 모델 학습·비교, 구간별 검증까지 전체 과정을 하나로 묶은 메인 파이프라인 | 박준석 (junseok) |
| `09_preprocessing_report.py` | `08_full_pipeline.py`에서 수행한 전처리 과정을 정리한 HTML 보고서(`preprocessing_report.html`) 생성 | 박준석 (junseok) |
| `10_make_docx_report.py` | `08_full_pipeline.py` 실행 결과를 읽어 이 폴더의 최종 docx 보고서를 생성 | 박준석 (junseok) |
| `config.py` | 컬럼 분류·유틸리티 함수 등 파이프라인 공통 설정 모듈 (팀 공용 설정. 각 파이프라인이 독립 실행되도록 같은 파일을 폴더마다 복사해 뒀습니다 — 7곳 모두 내용이 동일합니다) | 팀 공용 (원 작성자 미상) |

## 실행 순서

`01 → 02 → ... → 10` 순서가 실험을 처음부터 재현하는 순서입니다. `08_full_pipeline.py`는 01~06의 실험 결과를 정리해 만든 최종 파이프라인이므로, `08 → 09 → 10`만 실행해도 이 폴더의 최종 산출물(보고서)을 재현할 수 있습니다.

각 스크립트는 원천 데이터 `data/raw/race_entries.csv.gz`를 입력으로 씁니다.
경로는 `config.py`의 `RAW_ENTRIES` 상수에 들어 있습니다.
