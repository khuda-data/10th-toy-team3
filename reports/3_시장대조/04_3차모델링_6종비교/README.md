# 3차 모델링 — 모델 6종 비교

프로젝트 흐름 **3단계**. 배당률에서 뽑은 시장 확률을 어떻게 다루느냐에 따라 모델 6종을 만들어 비교하고,
가장 나은 LightGBM을 경주 단위 5-fold로 다시 검증한 결과입니다.

작성 이종원 (`happy97908a-cmyk`)

---

## 먼저 볼 것

`v5_final_model_report.html` 하나에 전체 보고서와 그래프가 전부 들어 있습니다.
그래프가 HTML 안에 박혀 있어서 인터넷 연결도, 별도 이미지 파일도, 파이썬 환경도 필요 없습니다.

> GitHub에서는 HTML이 소스 코드로 보입니다. 내려받아 브라우저로 열거나,
> 파일 화면의 **Raw** 버튼 주소를 `htmlpreview.github.io` 에 붙여 여시면 됩니다.

---

## 파일

파일명이 영문인 것은 생성 코드가 이 이름 그대로 쓰기 때문입니다. 아래가 한글 대응입니다.

| 파일 | 무엇인가 |
|---|---|
| `v5_final_model_report.html` | 최종 보고서 전문 + 모든 시각자료 |
| `v5_model_metrics.csv` | 모델 6종의 공통 최종 지표 (ROC-AUC · ROI 등) |
| `v5_model_detail_metrics.csv` | 위를 train · valid · test로 나눈 상세 지표 |
| `v5_general_model_threshold_comparison.csv` | EDGE 상위 10% · 20% · 30%로 잘랐을 때의 비교 |
| `v5_feature_importance_consensus.csv` | 6종 모델이 공통으로 중요하다고 본 피처 |
| `v5_lightgbm_5fold_results.csv` | LightGBM 경주 단위 5-fold 각각의 결과 |
| `v5_lightgbm_5fold_summary.json` | 위 5-fold의 평균 · 표준편차 · OOF 요약 |

`v5`는 이 실험에서 쓴 데이터셋 판번호입니다. `data/전처리_데이터셋/`의 v1~v8과는 다른 체계입니다.

---

## 결과 요약

| 순위 | 모델 | test ROC-AUC | 상위 10% ROI |
|---|---|---:|---:|
| 1 | Deep listwise ensemble | 0.7710 | −8.6% |
| 2 | Plackett-Luce hybrid | 0.7684 | +23.3% |
| 3 | LightGBM rank+binary | 0.7552 | +45.0% |
| 4 | CatBoost ordered | 0.7508 | +13.3% |
| 5 | Random Forest | 0.7507 | +55.0% |
| 6 | XGBoost | 0.7419 | +55.0% |

**ROC-AUC 순위와 수익 순위가 뒤집힙니다.** 가장 잘 맞히는 모델이 가장 많이 버는 모델은 아니었습니다.
다만 베팅 표본이 64건뿐이라 ROI 수치는 아직 흔들립니다.

여섯 모델 모두 시장의 ROC-AUC 0.817에는 닿지 못했습니다.

---

## 저장소에 없는 것

학습 데이터, 저장된 모델 파일(`models/`), 개별 PNG, OOF 행 단위 예측은 용량이 커서 제외했습니다.
보고서 열람에는 필요하지 않습니다.
