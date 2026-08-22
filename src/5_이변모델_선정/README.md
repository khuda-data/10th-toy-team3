# 이변모델 선정 코드

`reports/5_이변모델_선정/모델선정.docx`(RandomForest, Lift@10% 1.72배 [1.41, 2.02])로 이어지는 개선·선정 과정의 코드입니다. `박준석/이변_예측모델_완결_보고서.docx`, `박준석/이변_예측모델_최종전략_보고서.docx`가 이 코드의 산출물입니다.

## 코드 설명

| 파일 | 설명 | 작성자 |
|---|---|---|
| `11_improvement_experiments.py` | 이변 모델 6단계 개선 실험 — place 타겟+plcOdds ROI, min_samples_leaf 튜닝, 고배당 제거 로버스트니스, 배당 구간 필터, 피처 추가(hr_trend_3·jk_recent_form), 붕괴+다크호스 조합 | 박준석 (junseok) |
| `12_final_strategy.py` | 최종 전략 — 기존 C모델(단승) 결과에 winOdds 구간별 필터를 적용해 ROI 계산 | 박준석 (junseok) |
| `13_final_report_docx.py` | `12_final_strategy.py` 결과를 바탕으로 최종 전략 상세 보고서(docx) 생성 | 박준석 (junseok) |
| `14_upset_insight_and_allocation.py` | Part A: 이변마 vs 비이변마 피처 비교, K-means 군집화, 의사결정나무로 이변 규칙 추출, 배당 구간별 이변마 프로필 / Part B: Flat·기대값비례·구간별차등·Top-only 배팅 배분 전략을 valid에서 탐색해 test에서 검증 | 박준석 (junseok) |
| `15_complete_report.py` | 모델 비교(A/B/C)·배당 구간 필터 전략·이변마 공통점 분석·배분 전략·신뢰구간을 모두 종합한 최종 완결 보고서(docx) 생성 | 박준석 (junseok) |
| `config.py` | 컬럼 분류·유틸리티 함수 등 파이프라인 공통 설정 모듈 (팀 공용 `src/1_전처리/config.py`를 이 파이프라인에서 그대로 import해 사용) | 팀 공용 (원 작성자 미상) |

## 실행 순서

`11 → 12 → 13 → 14 → 15` 순서입니다. `08_full_pipeline.py`(레포의 `src/4_이변모델/`)로 만든 C모델 결과가 먼저 필요합니다. `../이변모델_선정_결과/`가 11·12·14의 산출물이고, `../이변_예측모델_완결_보고서.docx`·`../이변_예측모델_최종전략_보고서.docx`가 각각 15·13의 산출물입니다.
