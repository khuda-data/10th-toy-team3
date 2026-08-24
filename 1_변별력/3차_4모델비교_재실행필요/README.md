# 3차 4모델 비교 — 재실행 필요

최종 발표는 LightGBM·CatBoost·Random Forest·XGBoost를 같은 비시장 피처로 비교하는 구성을 채택했습니다. 다만 `main`의 기존 v5 4모델 산출물은 이 발표 전제를 충족하지 않아 최종 수치로 사용할 수 없습니다.

## 기존 v5 산출물을 제외한 이유

- 실제 학습 manifest의 128개 피처에 당일 `winAmt`, `plcAmt`, `totalAmt`, `log_winAmt`, `liq_per_horse`가 포함돼 있었습니다.
- 따라서 “당일 배당률과 발매금액 등 시장 정보를 제외한 독립 모델”의 근거가 아닙니다.
- 요약 보고서의 모델별 ROC-AUC와 개별 `metrics.json`도 서로 다른 실행값이어서 한 표로 인용하면 안 됩니다.
- Deep Listwise와 Plackett–Luce는 발표 모델에서 제외됐습니다.

## 확정해야 할 재실행 계약

1. 후보 데이터는 `data/전처리_데이터셋/v1_base/`의 동일한 시간순 Train·Valid·Test를 사용합니다.
2. 당일 시장 변수 7개(`winAmt`, `plcAmt`, `totalAmt`, `log_winAmt`, `liq_per_horse`, `gap_h`, `gap_d`)와 타깃·분할 식별자를 먼저 제외합니다.
3. 전처리기는 Train에만 적합하고 네 모델에 같은 원천 피처 계약을 적용합니다.
4. 모델·하이퍼파라미터·정규화·평가 코드를 Valid에서 고정한 뒤 Test를 한 번만 평가합니다.
5. 결과물에는 모델별 Test ROC-AUC, 시장 기준선, 피처 manifest, 실행 환경과 해시를 함께 저장합니다.

현재 `master`에서는 네 모델을 비교했다는 발표 구성은 남기되, 깨끗한 재실행 전까지 기존 v5의 정확한 AUC 숫자나 그래프를 발표 근거로 사용하지 않습니다.
