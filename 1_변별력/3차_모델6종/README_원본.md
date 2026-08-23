# Revised_v5 학습 데이터 및 모델

## 연구 목적

Revised_v5는 `winOdds` 같은 명시적 배당률을 모델 입력 피처로 넣지 않고, 말·기수·조교사·거리·부담중량·훈련·전개 관련 정보로 시장보다 나은 예측을 만들려 했던 초기 실험이다.

## 반드시 확인할 한계

사후 파일 감사 결과 실제 128개 학습 피처에는 `winAmt`, `plcAmt`, `totalAmt`, `log_winAmt`, `liq_per_horse`가 포함돼 있다. 따라서 이 모델을 시장정보가 완전히 제거된 깨끗한 무배당 모델로 해석하면 안 된다. 이 폴더는 당시 실행을 그대로 보존한 역사적 재현 패키지이며, 현재 배치 가능한 모델 근거가 아니다.

## 포함 내용

- `data/revised_v5`: 당시 train/valid/test 원본 수정 데이터와 무결성 보고서
- `data/revised_v5_preprocessed`: 수치화·결측 처리·스케일링된 학습 데이터, metadata, 전처리기와 검증 manifest
- `models/revised_v5_preprocessed_full`: Random Forest, XGBoost, LightGBM, CatBoost, Deep listwise ensemble, Plackett-Luce 실제 모델 파일과 metrics, feature importance, test predictions
- `reports`: 당시 v5 HTML·PDF·Markdown 보고서와 지표·시각화
- `logs`: 보존된 v5 학습 실행 로그
- `src`: v5 데이터 준비·전처리·6모델 학습·검증·보고서 생성 스크립트

원본 Revised_v5 train/valid/test와 활성 작업본의 SHA-256은 각각 일치한다. 패키지 생성 시 데이터·모델 원본과 복사본을 SHA-256으로 다시 대조한다.
