# 이변모델 선정 코드

`reports/5_이변모델_선정/01_모델선정.docx`(RandomForest, Lift@10% 1.72배 [1.41, 2.02])로 이어지는 개선·선정 과정의 코드입니다. `reports/5_이변모델_선정/03_종합보고서.docx`, `reports/5_이변모델_선정/02_배당구간_필터전략.docx`가 이 코드의 산출물입니다.

## 코드 설명

| 파일 | 설명 | 작성자 |
|---|---|---|
| `01_improvement_experiments.py` | 이변 모델 6단계 개선 실험 — place 타겟+plcOdds ROI, min_samples_leaf 튜닝, 고배당 제거 로버스트니스, 배당 구간 필터, 피처 추가(hr_trend_3·jk_recent_form), 붕괴+다크호스 조합 | 박준석 (junseok) |
| `02_final_strategy.py` | 최종 전략 — 기존 C모델(단승) 결과에 winOdds 구간별 필터를 적용해 ROI 계산 | 박준석 (junseok) |
| `03_final_report_docx.py` | `02_final_strategy.py` 결과를 바탕으로 최종 전략 상세 보고서(docx) 생성 | 박준석 (junseok) |
| `04_upset_insight_and_allocation.py` | Part A: 이변마 vs 비이변마 피처 비교, K-means 군집화, 의사결정나무로 이변 규칙 추출, 배당 구간별 이변마 프로필 / Part B: Flat·기대값비례·구간별차등·Top-only 배팅 배분 전략을 valid에서 탐색해 test에서 검증 | 박준석 (junseok) |
| `05_complete_report.py` | 모델 비교(A/B/C)·배당 구간 필터 전략·이변마 공통점 분석·배분 전략·신뢰구간을 모두 종합한 최종 완결 보고서(docx) 생성 | 박준석 (junseok) |
| `config.py` | 컬럼 분류·유틸리티 함수 등 파이프라인 공통 설정 모듈 (팀 공용 설정. 각 파이프라인이 독립 실행되도록 같은 파일을 폴더마다 복사해 뒀습니다 — 7곳 모두 내용이 동일합니다) | 팀 공용 (원 작성자 미상) |

## 실행 순서

`01 → 02 → 03 → 04 → 05` 순서입니다. [`src/4_이변모델/08_full_pipeline.py`](../4_이변모델/08_full_pipeline.py)로 만든 C모델 결과가 먼저 필요합니다. `reports/5_이변모델_선정/결과/`가 11·12·14의 산출물이고, `reports/5_이변모델_선정/03_종합보고서.docx`·`reports/5_이변모델_선정/02_배당구간_필터전략.docx`가 각각 15·13의 산출물입니다.
