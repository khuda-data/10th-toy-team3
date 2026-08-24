# 찬진 최종 이변 모델

파트 2의 공식 기존 Test 결과를 만드는 잠금 평가 패키지입니다.

## 분석 대상

| 타깃 | 후보군 | 정답 |
|---|---|---|
| 다크호스 | `pop_pct >= 0.50`, 인기 하위 50% | `place == 1`, 3착 이내 입상 |
| 인기마 부진 | `pop_pct <= 0.25`, 인기 상위 25% | `fin_pct >= 0.50`, 착순 하위 50% |

저장된 `upset_B`와 다크호스 재계산 라벨, `upset_A`와 인기마 부진 재계산 라벨은 각각 100% 일치합니다.

## 데이터 경로

- 전체 원천: `../../data/race_entries.csv.gz`
- 시간순 분할: `../../data/전처리_데이터셋/v1_base/`
- train 19,637행 / valid 6,591행 / test 6,660행

코드는 저장소 루트를 자동으로 찾도록 정리했습니다. 기존 `final.csv.gz`와 확장자 없는 split CSV를 요구하던 경로는 현재 `master` 구조에 맞게 수정했습니다.

## 검증 절차

1. train·valid 데이터와 라벨을 점검합니다.
2. Logistic Regression, Random Forest, XGBoost 후보를 valid에서 비교합니다.
3. valid Lift@10% 중심으로 모델을 선택합니다.
4. 선택 설정을 `configs/locked_config.json`에 잠급니다.
5. 잠금 이후 test를 한 번 평가합니다.

## 공식 다크호스 Core 결과

| 지표 | 값 |
|---|---:|
| Test 후보 | 3,528두 |
| Test 양성 | 478두 |
| 기준 입상률 | 13.5% |
| 상위 10% 입상률 | 24.7% |
| Lift@10% | 1.82 |
| ROC-AUC | 0.647 |
| PR-AUC | 0.213 |

## 해석

- 비인기마 중 실제 입상 가능성이 높은 후보를 압축하는 선별력은 확인됐습니다.
- 상위 10% 실현 ROI는 양수였지만 고배당 한 건을 제거하면 음수이고 95% 신뢰구간도 0을 포함합니다.
- 따라서 최종 결론은 **후보 선별력 확인·수익성 미확정**입니다.
- 피처 중요도는 연관 신호이며 시장의 인과적 실수로 해석하지 않습니다.

## 실행 순서

```bash
python scripts/01_preflight.py
python scripts/02_train_validate.py
python scripts/03_lock_config.py
python scripts/04_evaluate_test.py
python scripts/05_build_report.py
```

원본 결과는 `outputs/tables/`, 그래프는 `outputs/figures/`, 최종 해석은 `reports/final_upset_model_report.md`에 있습니다.
