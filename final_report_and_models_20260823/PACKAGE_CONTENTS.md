# 최종 보고서 및 최근 실제 학습 모델 패키지

이 폴더는 2026-08-23 현재 프로젝트의 권위 최종 보고서와 최근 실제 학습에서 저장된 모델 파일을 함께 보관한다.

## 최종 보고서

- `report/index.html`
- `report/index.pdf`
- `report/report_data_snapshot.json`
- `report/report_manifest.json`
- `report/html_validation.json`
- `report/pdf_validation.json`

위 파일은 로컬 권위 원본 `outputs/reports/final_comprehensive_research_report_20260823`에서 다시 동기화했다.

## 장기 연구 및 직전 작업 보고서 원본

`reports`에는 다음 범위를 원본 디렉터리 이름 그대로 보존한다.

- 10시간 이상 장기 작업의 본보고서 `market_probability_advantage_deep_research_20260823`
- 본보고서 `report_manifest.json`이 직접 인용한 실제 실험 결과, 예측, 로그, 독립 감사 파일
- 직전 작업의 서울·부경 v11 전체 착순 재학습 보고서 `revised_v11_seoul_bugyeong_full_rerun_20260822`
- 직전 중간보고서, Git-로컬 방향 비교, 전처리 비교, 시장 대비 베팅 검증, 8~15배 재검증, 확률 용어 설명, 프로젝트 학습교재
- 장기 작업 이후의 H10A, H11B, H11C, H11D, H14, H17 및 prospective power 결과

총 36개 보고서·실험 디렉터리와 1,299개 파일을 원본과 SHA-256으로 대조했으며 누락과 불일치는 각각 0개다.

## 전체 착순 모델

`models/revised_v11_seoul_bugyeong_full_rank_20260822`에 다음 6개 모델의 실제 저장 파라미터, manifest, metrics, feature importance, validation/test 예측을 포함한다.

1. Random Forest Rank
2. XGBoost Ranker
3. LightGBM LambdaRank
4. CatBoost YetiRank
5. Deep RankNet (3 seeds)
6. Plackett-Luce Full Rank

## 시장 특이점 및 후속 모델

8월 22~23일에 저장된 시장 앵커, 이변 피처, rolling-origin, place calibration, deep MLP, race attention, sparse deep residual, partial pooling, defensive experiment, H10A, H11B, H13, H14, H17 모델 디렉터리를 모두 포함한다.

총 모델 디렉터리 21개, 파일 159개를 원본과 SHA-256으로 대조했으며 복사 누락과 해시 불일치는 각각 0개다. 여기에는 장기 연구 모델과 그 직전 v11 전체 착순·시장 앵커·이변 모델이 함께 들어 있다.

## 해석 주의

모델 파일의 존재는 실제 학습·저장 근거이지만 배치 가능한 수익을 뜻하지 않는다. 최종 보고서의 현재 결론은 확정된 실전 수익 전략 없음, 권고 실베팅 금액 0이다.
