# revised_v5 이후 데이터 전처리 개선 보고서

## 최종 상태

종단 검증 결과는 **PASS**다.

| 검사 항목 | Train | Validation | Test |
|---|---:|---:|---:|
| 원본 행 수 | 19,617 | 6,582 | 6,639 |
| 전처리 후 행 수 | 19,617 | 6,582 | 6,639 |
| 전체 열 숫자형 | PASS | PASS | PASS |
| NaN | 0 | 0 | 0 |
| 무한값 | 0 | 0 | 0 |
| `entry_id` 코드 고유성 | PASS | PASS | PASS |
| 원본 타깃 정렬 보존 | PASS | PASS | PASS |
| 원본 재변환과 저장값 일치 | PASS | PASS | PASS |

## 확인된 원본 문제

- `revised_v5`는 146개 열이며 분할마다 일부 열의 dtype이 달랐다.
- `jkNo`는 train/valid에서 문자열, test에서 숫자형이었다.
- `owNo`와 `trNo`도 분할에 따라 문자열/숫자형이 달랐다.
- 모델 후보 121개 중 `ageCond`만 실제 문자열 피처였다.
- 기존 축소 데이터의 42개 모델 피처는 숫자형이었지만 학습 기준 스케일링이 적용되지 않았다.
- 기존 학습 코드에서는 연속형 `hr_style`을 범주형 코드로 다시 변환했다. 새 파이프라인에서는 숫자형 연속 피처로 처리한다.

## 적용한 전처리

1. train/valid/test 스키마와 열 순서를 검사했다.
2. 중복 행과 중복 `entry_id`, 이진 타깃 조건을 검사했다.
3. 누수 열과 식별자 열을 모델 피처에서 제외했다.
4. 120개 숫자 피처를 엄격하게 숫자로 변환했다. 변환 불가능 값이 있으면 즉시 실패하도록 했다.
5. 무한값을 결측값으로 치환하도록 정의했다.
6. 숫자 피처에 train 기준 median imputation을 적용했다.
7. 숫자 피처에 train 기준 StandardScaler를 적용했다.
8. `ageCond`에는 train 기준 최빈값 대치와 one-hot encoding을 적용했다. 미등록 범주는 오류 없이 0 벡터로 처리한다.
9. `race_id`, `entry_id`, `hrName`은 추적용 숫자 코드로 변환했지만 모델 피처에서는 제외했다.
10. 원본 문자열 식별자는 EDGE 시장 데이터 연결을 위해 별도 metadata 파일로 보존했다.
11. 전처리 후 숫자형, NaN, 무한값, 행 수, 키 고유성, 스케일 통계와 재변환 일치를 검사했다.

전처리기는 **train 데이터에만 적합**했고 valid/test에는 변환만 적용했다.

## 출력 구조

### 전체 revised_v5

`data/revised_v5_preprocessed/`

- 숫자 원천 피처: 120개
- `ageCond` one-hot 피처: 8개
- 최종 모델 피처: 128개
- train 숫자 피처 최대 절대 평균: `1.200e-15`
- train 숫자 피처 표준편차의 최대 오차: `4.441e-16`

### EDGE 0.5% 축소 데이터

`data/edge_top_005_preprocessed/`

- 최종 모델 피처: 42개
- 전체 전처리 데이터에서 동일 열을 파생
- CSV 재저장에 따른 최대 절대 반올림 차이: `8.881784197001252e-16`
- 허용오차 `1e-15`에서 세 분할 모두 일치

## 실제 학습·예측 테스트

전처리 완료된 42개 피처를 추가 형변환 없이 직접 읽어 LambdaRank와 binary LightGBM 모델을 새로 학습했다.

| 항목 | 결과 |
|---|---:|
| 모델 입력 피처 | 42 |
| 테스트 예측 행 | 6,639 |
| 유한값 예측 | 6,639 / 6,639 |
| EDGE 저장 행 | 6,639 |
| Test ROC-AUC | 0.7618233514 |
| Test 1위 적중률 | 0.3181102362 |
| 상위 10% EDGE ROI | -0.0250 |

ROI는 전처리 성공 여부를 판정하는 지표가 아니다. 데이터·모델 종단 검증은 모든 구조 및 유한값 검사 통과 여부로 판정했으며 결과는 PASS다.

## 근거 파일

- `data/revised_v5_preprocessed/preprocessing_validation.json`
- `data/edge_top_005_preprocessed/preprocessing_validation.json`
- `outputs/reports/preprocessing/end_to_end_validation.json`
- `outputs/reports/preprocessing/revised_v5_preprocessing.log`
- `outputs/reports/preprocessing/edge_subset_preprocessing.log`
- `outputs/reports/preprocessing/preprocessed_model_training.log`
- `outputs/reports/preprocessing/end_to_end_validation.log`
