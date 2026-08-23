# 시장 앵커 모델 재학습·ΔLL·7개 승식 검증 보고서

## 지표 개념과 산식

- **시장 앵커**: 각 말의 점수를 `ηᵢ = ln(qᵢ) + f(xᵢ)`로 둔다. `ln(q)`의 계수는 1로 고정하고, 141개 펀더멘털 피처는 시장이 남긴 오차 `f(x)`만 학습한다. 배당은 일반 입력 피처로 넣지 않았다.
- **로그우도(LL)**: 실제 승자의 예측확률 로그를 경주별로 합한 값이다. 확률 예측이 정확할수록 0에 가까워진다.
- **ΔLL/경주**: `(LL_앵커모델 - LL_시장) / 경주 수`. 양수면 동일 경주에서 시장확률보다 추가 예측정보가 있다는 뜻이다.
- **EDGE**: 모델 사건확률에서 같은 사건의 시장확률을 뺀 값이다. 단승은 직접 `q`, 연승은 직접 `q_plc`, 복합승식은 `q`의 Plackett–Luce/Harville 확장값을 비교 기준으로 썼다.
- **ROI**: `(총 환급 - 총 베팅액) / 총 베팅액`. 공식 최종 배당은 공제 후 값이므로 단승 20%, 다중 27%를 다시 차감하지 않았다.
- **적중률**: 선택한 베팅 중 실제로 적중한 비율이다.
- **MDD**: 자산곡선의 이전 최고점 대비 최대 하락률이다. 여기서는 베팅마다 자본의 1%를 고정 투입한 비교 곡선이다.
- **Sharpe**: 베팅 수익의 평균을 표준편차로 나눈 뒤 베팅 수 제곱근을 곱한 안정성 지표다.
- **Fractional Kelly**: 단승·연승의 선택별 배당과 모델확률로 full Kelly의 1/4을 쓰고 1회 2%로 제한했다.

## 실행 설계와 무결성

- 데이터: `C:\Users\user\source\repos\PythonApplication2\PythonApplication2\data\revised_v11_seoul_bugyeong_rank_clean_preprocessed`의 141개 수치형 피처.
- 시간 분할: train `20230805~20250511`, validation `20250517~20251227`, test `20251228~20260809`. 순서 위반은 없었다.
- 시장 원본: `C:\Users\user\Downloads\final (2).csv.gz`, SHA-256 `964BD9A7AB7E36247FC5A7E5FFD04F9EC02439491B1DACAEF7918EB0AAE80195`.
- `q`는 경주별 정규화 `1 / winOdds`와 최대 절대오차 `2.776e-16`로 일치했다.
- 모델/하이퍼파라미터와 EDGE 정책은 validation에서 결정했다. test는 최종 1회 평가에만 사용했다.
- 조건부 로짓 L2: `100.0`; Base Margin 트리 수: `188`.
- 확률합 최대오차: `4.441e-16`.
- 검증: 21개 검사 모두 PASS.

## test 시장 대비 ΔLL

| model | races | delta_ll_per_race | delta_ll_ci_low_daily_block | delta_ll_ci_high_daily_block | delta_ll_confirmed_positive_95pct | model_log_loss_per_race | market_log_loss_per_race | model_top1_hit_rate | market_top1_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 시장 앵커 조건부 로짓 | 1101 | 0.013974 | 0.002318 | 0.025063 | True | 1.779720 | 1.793694 | 38.33% | 37.87% |
| 시장 앵커 Base Margin | 1101 | 0.013083 | 0.006614 | 0.019647 | True | 1.780611 | 1.793694 | 37.51% | 37.87% |
| 두 시장 앵커 앙상블 | 1101 | 0.017194 | 0.009507 | 0.025014 | True | 1.776500 | 1.793694 | 37.60% | 37.87% |

95% CI는 경주일 블록 5,000회 부트스트랩이다. CI 하한이 0보다 큰 경우에만 시장보다 정보량이 유의하게 개선됐다고 판정한다.

### 과거 Claude 협업 시기 시장 앵커 근거와의 위치 비교

| run | protocol | races | delta_ll_per_race | ci_low | ci_high | source |
| --- | --- | --- | --- | --- | --- | --- |
| Claude-era conditional logit walk-forward | expanding-window OOS | 4009 | +0.004906 | +0.001256 | +0.008635 | benter_market_anchored_20260820/walkforward_summary.json |
| Claude-era Base Margin walk-forward | expanding-window OOS | 4009 | +0.007392 | +0.004373 | +0.010444 | base_margin_boosting_20260821/summary.json |
| 시장 앵커 조건부 로짓 | fixed latest test | 1101 | +0.013974 | +0.002318 | +0.025063 | market_anchor_same_test_20260822/delta_ll_metrics.csv |
| 시장 앵커 Base Margin | fixed latest test | 1101 | +0.013083 | +0.006614 | +0.019647 | market_anchor_same_test_20260822/delta_ll_metrics.csv |
| 두 시장 앵커 앙상블 | fixed latest test | 1101 | +0.017194 | +0.009507 | +0.025014 | market_anchor_same_test_20260822/delta_ll_metrics.csv |

과거 두 행은 4,009경주 확장 윈도우 워크포워드이고, 새 행은 고정된 최신 635경주 test다. 표본·피처·학습 규약이 다르므로 새 수치가 더 크다는 이유만으로 모델이 개선됐다고 단정하지 않는다. 이 표는 재현 위치를 보여주는 비교이지 동일조건 우열 검정이 아니다.

## validation 잠금 후 test 7개 승식

| bet_type | model | selection_policy | test_bets | test_wins | test_hit_rate | test_roi | test_roi_ci_low_daily_block | test_roi_ci_high_daily_block | profit_confirmed_95pct | average_selected_odds | average_winning_dividend | flat_1pct_mdd | per_bet_sharpe | kelly_status | kelly_total_return | kelly_mdd | kelly_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 단승 | 시장 앵커 Base Margin | top_10pct_edge | 108 | 45 | +41.67% | -14.54% | -35.54% | +6.77% | False | 2.298 | 2.051 | -17.23% | -1.396 | AVAILABLE | -4.96% | -6.75% | -1.291 |
| 연승 | 시장 앵커 Base Margin | top_10pct_edge | 101 | 74 | +73.27% | -7.52% | -17.01% | +2.38% | False | 1.276 | 1.262 | -8.86% | -1.316 | AVAILABLE | -14.68% | -17.07% | -1.316 |
| 복승 | 두 시장 앵커 앙상블 | top_10pct_edge | 108 | 34 | +31.48% | +39.72% | -3.30% | +84.45% | False | N/A | 4.438 | -12.63% | 1.785 | UNAVAILABLE_FOR_LOSING_COMBINATIONS | N/A | N/A | N/A |
| 쌍승 | 두 시장 앵커 앙상블 | top_10pct_edge | 108 | 21 | +19.44% | +16.76% | -31.05% | +67.62% | False | N/A | 6.005 | -21.35% | 0.643 | UNAVAILABLE_FOR_LOSING_COMBINATIONS | N/A | N/A | N/A |
| 삼복승 | 시장 앵커 Base Margin | top_20pct_edge | 216 | 34 | +15.74% | -7.69% | -39.63% | +26.52% | False | N/A | 5.865 | -39.33% | -0.463 | UNAVAILABLE_FOR_LOSING_COMBINATIONS | N/A | N/A | N/A |
| 삼쌍승 | 시장 앵커 Base Margin | top_10pct_edge | 108 | 5 | +4.63% | -32.96% | -86.74% | +32.35% | False | N/A | 14.480 | -43.27% | -1.014 | UNAVAILABLE_FOR_LOSING_COMBINATIONS | N/A | N/A | N/A |
| 복연승 | 두 시장 앵커 앙상블 | top_10pct_edge | 108 | 52 | +48.15% | +15.93% | -5.85% | +39.13% | False | N/A | 2.408 | -12.12% | 1.261 | UNAVAILABLE_FOR_LOSING_COMBINATIONS | N/A | N/A | N/A |

각 승식마다 `모델 × EDGE 상위 10/20/30% × 양의 EDGE × 전체`를 validation에서 비교했고, 최소 60건 조건의 최상 정책 하나만 고정해 test에 적용했다. test에서 다시 좋은 셀을 고르지 않았다.

### 직전 비앵커 전체착순 모델과 같은 test 비교

| bet_type | anchor_model | anchor_policy | anchor_test_bets | anchor_test_roi | prior_full_rank_model | prior_full_rank_policy | prior_full_rank_test_bets | prior_full_rank_test_roi | roi_delta_anchor_minus_full_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 단승 | 시장 앵커 Base Margin | top_10pct_edge | 108 | -14.54% | Random Forest | top_10pct_edge | 108 | -3.06% | -11.48% |
| 연승 | 시장 앵커 Base Margin | top_10pct_edge | 101 | -7.52% | XGBoost Ranker | top_10pct_edge | 103 | +1.26% | -8.79% |
| 복승 | 두 시장 앵커 앙상블 | top_10pct_edge | 108 | +39.72% | Random Forest | positive_edge | 685 | -17.65% | +57.37% |
| 쌍승 | 두 시장 앵커 앙상블 | top_10pct_edge | 108 | +16.76% | Random Forest | positive_edge | 688 | -18.04% | +34.80% |
| 삼복승 | 시장 앵커 Base Margin | top_20pct_edge | 216 | -7.69% | XGBoost Ranker | top_30pct_edge | 324 | -24.54% | +16.85% |
| 삼쌍승 | 시장 앵커 Base Margin | top_10pct_edge | 108 | -32.96% | Random Forest | positive_edge | 752 | -66.52% | +33.55% |
| 복연승 | 두 시장 앵커 앙상블 | top_10pct_edge | 108 | +15.93% | LightGBM LambdaRank | top_10pct_edge | 108 | -53.61% | +69.54% |

두 실험 모두 validation에서 각자 정책을 잠갔지만 선택한 모델·정책과 베팅 수가 다르다. 따라서 `ROI 차이`는 동일 test의 기술적 비교이며, 정책 자체의 인과적 개선량이나 통계적 우월성 검정으로 해석하지 않는다.

## 같은 test의 시장 인기선택 기준선

| bet_type | races | hit_rate | roi |
| --- | --- | --- | --- |
| 단승 | 1077 | 37.70% | -17.60% |
| 연승 | 1004 | 66.83% | -15.34% |
| 복승 | 1077 | 19.87% | -8.46% |
| 쌍승 | 1077 | 11.61% | -13.90% |
| 삼복승 | 1077 | 11.14% | -24.60% |
| 삼쌍승 | 1077 | 3.06% | -33.49% |
| 복연승 | 1077 | 36.58% | -20.74% |

복합 승식의 ROI는 마사회 공식 적중 조합 최종 배당으로 계산했다. 다만 패배 조합의 사전 배당이 없으므로 복합 승식의 실제 시장 EDGE, 전체 선택의 평균 배당, Kelly 비중은 확정할 수 없다. 해당 항목은 `N/A`로 남겼다. 복합 사건확률은 Monte Carlo 대신 동일 Plackett–Luce 가정의 순열을 정확 합산해 시뮬레이션 오차를 제거했다.

## XAI: 시장 오차를 설명한 상위 피처

| model_key | feature | normalized_importance | method |
| --- | --- | --- | --- |
| anchored_base_margin | te_trName | 6.973% | sum of tree impurity importances |
| anchored_base_margin | clinic_30d | 6.250% | sum of tree impurity importances |
| anchored_base_margin | hr_winrate__z | 5.016% | sum of tree impurity importances |
| anchored_base_margin | jk_winrate__z | 3.876% | sum of tree impurity importances |
| anchored_base_margin | jk_winrate | 3.833% | sum of tree impurity importances |
| anchored_base_margin | tr_winrate__z | 3.385% | sum of tree impurity importances |
| anchored_base_margin | tr_plcrate | 3.277% | sum of tree impurity importances |
| anchored_base_margin | style_vs_race | 3.234% | sum of tree impurity importances |
| anchored_base_margin | te_jkName | 3.029% | sum of tree impurity importances |
| anchored_base_margin | jk_plcrate | 2.687% | sum of tree impurity importances |
| anchored_conditional_logit | tool_n | 4.270% | absolute standardized coefficient |
| anchored_conditional_logit | clinic_30d | 2.689% | absolute standardized coefficient |
| anchored_conditional_logit | train_runs_14 | 2.681% | absolute standardized coefficient |
| anchored_conditional_logit | jk_winrate | 2.673% | absolute standardized coefficient |
| anchored_conditional_logit | wgBudam | 2.600% | absolute standardized coefficient |
| anchored_conditional_logit | wg_diff__pr | 2.427% | absolute standardized coefficient |
| anchored_conditional_logit | tool_BlueBond편자 | 2.266% | absolute standardized coefficient |
| anchored_conditional_logit | tr_plcrate | 2.243% | absolute standardized coefficient |
| anchored_conditional_logit | age | 2.193% | absolute standardized coefficient |
| anchored_conditional_logit | wg | 2.138% | absolute standardized coefficient |

조건부 로짓 중요도는 전처리된 피처의 절대계수, Base Margin 중요도는 사용 트리의 impurity importance 합이다. 이는 인과효과가 아니라 시장확률 위에 추가된 예측 신호의 모델 내부 기여도다.

## 위험 경고와 결론

test ROI 30% 이상 셀이 있어 누수·배당 이상·정책 과적합 경고를 발동했다: 복승

ΔLL의 95% CI와 승식별 ROI CI는 서로 다른 질문이다. ΔLL이 유의해도 마사회 공제와 배당 분산을 이긴 수익이 자동으로 확인되지는 않는다. 또한 이 실험은 파일에 저장된 **최종 배당**을 앵커로 사용하므로, 실제 배포 전에는 베팅 시점 스냅샷 배당으로 동일 검증을 다시 해야 한다.
