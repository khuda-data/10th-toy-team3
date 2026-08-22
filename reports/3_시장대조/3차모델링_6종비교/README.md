# revised_v5 보고서 제출본

이 폴더는 GitHub 제출용으로 필요한 결과만 모은 경량 패키지다.
학습 데이터, 저장 모델, 원본 이미지 파일은 포함하지 않는다.

## 열람 방법

`v5_final_model_report.html`을 웹 브라우저에서 열면 된다. 그래프가 HTML 파일 안에 포함되어 있으므로 인터넷 연결, 별도 이미지 파일, Python 환경이 필요하지 않다.

## 포함 파일

- `v5_final_model_report.html`: 전체 최종 보고서와 모든 시각자료
- `v5_model_metrics.csv`: 6개 일반 모델의 공통 최종 지표
- `v5_model_detail_metrics.csv`: Train·Validation·Test 상세 지표
- `v5_general_model_threshold_comparison.csv`: EDGE 상위 10%·20%·30% 선택 범위 비교
- `v5_feature_importance_consensus.csv`: 6개 모델 합의형 피처 중요도
- `v5_lightgbm_5fold_results.csv`: 경주 단위 5-fold별 결과
- `v5_lightgbm_5fold_summary.json`: 5-fold 평균·표준편차·OOF 요약

## 제외한 파일

`data/`, `models/`, 학습 중간 산출물, 개별 PNG 그래프, OOF 행 단위 예측 파일은 용량이 크거나 보고서 열람에 필요하지 않아 제외했다. 보고서의 그래프는 HTML에 내장되어 있다.

## GitHub에 올릴 때

저장소 전체가 아니라 이 폴더만 추가하면 된다.

```powershell
git add report_submission
git commit -m "Add revised_v5 final report submission"
```
