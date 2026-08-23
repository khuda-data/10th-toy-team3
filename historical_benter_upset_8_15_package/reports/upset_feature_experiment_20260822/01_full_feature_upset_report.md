# 감사 후 전체 133피처 이변 말 탐색 및 수익성 보고서

## 지표 개념과 산식

- **이변 점수**: `모델의 연승권 진입확률 - 시장 q_plc`. 시장 인기 하위 50% 말 중 이 값이 가장 큰 한 마리를 경주별 후보로 삼는다.
- **이변 피처 순열 중요도**: train 내부 마지막 20%에서 한 피처를 교란했을 때 상위 10·20·30·40% 이변 후보의 평균 실현 EDGE(`place-q_plc`)가 얼마나 감소하는지 나타낸다.
- **정규화 중요도 가중치**: 양의 순열 중요도를 합계 1로 정규화한 비중이다.
- **ΔLL/경주**: `(모델 LL-시장 LL)/경주 수`. 양수일수록 시장보다 확률분포가 정확하다.
- **EDGE**: 같은 사건에 대한 `모델확률-시장확률`이다.
- **ROI**: `(총 환급-총 베팅액)/총 베팅액`이다. 공식 최종 배당에는 공제가 반영돼 있어 20%·27%를 다시 차감하지 않는다.
- **Hit Rate**: 선택 베팅 중 적중 비율이다.
- **MDD**: 누적자산의 이전 최고점 대비 최대 하락률이다.
- **Sharpe**: 평균 수익을 수익 변동성으로 나눈 안정성 지표다.
- **95% CI**: 경주일 블록 5,000회 부트스트랩 구간이며, 하한이 0보다 클 때만 수익·정보 개선을 확인한다.
- **FDR q-value**: 56개 ROI 비교의 우연한 양성을 통제하는 Benjamini-Hochberg 보정값이다. `q<=0.05`와 CI 하한 양수를 함께 만족해야 기계적 통계 통과로 표시한다.

## 사전 고정 설계

- 프로젝트 원본에서 정확히 재현된 `upset_B=(pop_pct>=0.5 AND place=1)`를 양성 이변 정의로 사용했다. `upset_B` 자체는 사후 라벨이므로 입력에는 넣지 않았다.
- 각 경주에서 시장 인기 하위 50% 중 양의 연승 EDGE를 가진 말 가운데 이변 점수가 가장 높은 한 마리만 후보로 삼았다.
- 상위 10·20·30·40% 비율 자체를 사전에 고정했다. 각 split의 이변 점수만 내림차순 정렬해 정확히 `ceil(후보수×비율)`개를 골랐으며 결과 라벨은 경계 계산에 쓰지 않았다.
- 피처 중요도는 train을 앞 70% 학습·다음 10% 트리수 조정·마지막 20% 중요도 평가로 나눠 산출했다. validation과 test 결과는 피처 선별에 사용하지 않았다.
- 피처 재학습 임계값은 **정규화된 양의 순열 중요도 0.5% 이상**인 운영 기준이며 p-value가 아니다. 선택된 두 피처의 중요도 CI는 모두 0을 포함했다.
- 선택 피처는 2개이며 목록은 `selected_upset_features.json`에 저장했다.
- 배당은 `ln(q)` 고정 offset과 평가에만 사용했고 일반 피처행렬에는 넣지 않았다. 최종 발매총액인 `winAmt·plcAmt·totalAmt·liq_per_horse`도 제외해 133피처를 사용했다.
- 복합 승식은 이변 말 한 마리를 반드시 포함하는 모든 가능한 티켓 중 모델 사건확률이 최대인 티켓을 택했다. 복합 시장확률은 q 기반 PL/Harville proxy이며 실제 풀 확률은 아니다.
- 재학습 타깃은 승자(`win`)이며 연승확률은 승리확률에서 Plackett-Luce로 파생했다. `upset_B` 직접 분류모델은 아니다.

## 이변 피처 중요도 상위 30개

| rank | feature | positive_importance_weight | mean_upset_tail_realized_edge_drop | ci_low_daily_block | ci_high_daily_block | selected_for_retraining | ci_confirmed_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | style_vs_race | 93.684% | +0.001358 | -0.000694 | +0.004345 | True | False |
| 2 | jk_winrate__pr | 6.316% | +0.000092 | -0.009732 | +0.009990 | True | False |
| 3 | age | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 4 | ageCond_2세 | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 5 | ageCond_3세 | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 6 | ageCond_3세이상 | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 7 | ageCond_4세이상 | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 8 | ageCond_남3/북2 | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 9 | ageCond_남3/북3 | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 10 | ageCond_남4이하/북3 | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 11 | age__z__missing | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 12 | buga1 | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 13 | chulNo | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 14 | clinic_30d | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 15 | hr_dist_chg | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 16 | hr_dist_starts | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 17 | hr_dist_winrate | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 18 | hr_last_dist | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 19 | hr_last_finpct | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 20 | hr_last_finpct__missing | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 21 | hr_last_ord | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 22 | hr_plcrate | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 23 | hr_prev_rating | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 24 | hr_prev_rating__missing | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 25 | hr_rest_days | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 26 | hr_rest_days__pr | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 27 | hr_rest_days__pr__missing | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 28 | hr_style | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 29 | hr_style_n | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |
| 30 | hr_style_sd | 0.000% | +0.000000 | +0.000000 | +0.000000 | False | False |

## split별 정확 상위 비율과 점수 경계

| split | model_key | model | fraction | fraction_label | score_threshold | candidate_pool | selected_target_count | actual_selected_fraction | selection_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 10% | top_10pct | 0.05720989 | 336 | 34 | 10.12% | within-split score rank; outcomes unused |
| valid | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 20% | top_20pct | 0.03638842 | 336 | 68 | 20.24% | within-split score rank; outcomes unused |
| valid | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 30% | top_30pct | 0.02663936 | 336 | 101 | 30.06% | within-split score rank; outcomes unused |
| valid | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 40% | top_40pct | 0.02141783 | 336 | 135 | 40.18% | within-split score rank; outcomes unused |
| test | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 10% | top_10pct | 0.03983335 | 282 | 29 | 10.28% | within-split score rank; outcomes unused |
| test | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 20% | top_20pct | 0.02764377 | 282 | 57 | 20.21% | within-split score rank; outcomes unused |
| test | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 30% | top_30pct | 0.02026084 | 282 | 85 | 30.14% | within-split score rank; outcomes unused |
| test | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 40% | top_40pct | 0.01643284 | 282 | 113 | 40.07% | within-split score rank; outcomes unused |

## test 이변 말 구성과 실제 적중

| split | model_key | model | fraction | fraction_label | candidate_pool | selected_horses | actual_selected_fraction | mean_market_rank | mean_win_odds | actual_win_rate | actual_place_rate | stored_upset_B_rate | exact_top3_upset_rate | market_rank_2_3 | market_rank_4_6 | market_rank_7_plus | win_odds_lt_10 | win_odds_10_20 | win_odds_20_30 | win_odds_30_plus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 0.1 | top_10pct | 282 | 29 | 10.28% | 6.620689655172414 | 19.717241379310348 | 17.24% | 24.14% | 24.14% | 24.14% | 0 | 15 | 14 | 4 | 12 | 9 | 4 |
| test | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 0.2 | top_20pct | 282 | 57 | 20.21% | 7.0 | 22.614035087719298 | 12.28% | 21.05% | 21.05% | 21.05% | 0 | 23 | 34 | 5 | 23 | 18 | 11 |
| test | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 0.3 | top_30pct | 282 | 85 | 30.14% | 7.4 | 27.669411764705877 | 8.24% | 17.65% | 17.65% | 17.65% | 0 | 30 | 55 | 7 | 28 | 24 | 26 |
| test | full_upset_base_margin | 감사 후 전체 133피처 이변 Base Margin | 0.4 | top_40pct | 282 | 113 | 40.07% | 7.513274336283186 | 28.5787610619469 | 6.19% | 13.27% | 13.27% | 13.27% | 0 | 39 | 74 | 7 | 34 | 36 | 36 |

## 전체 피처 모델 test ΔLL

| model | races | delta_ll_per_race | delta_ll_ci_low_daily_block | delta_ll_ci_high_daily_block | delta_ll_confirmed_positive_95pct | model_top1_hit_rate | market_top1_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 감사 후 전체 133피처 이변 Base Margin | 614 | +0.015399 | +0.007217 | +0.023289 | True | 36.97% | 37.13% |

## 전체 피처 모델 test 승식별 ROI

| 승식 | top_10pct | top_20pct | top_30pct | top_40pct |
| --- | --- | --- | --- | --- |
| 단승 | +112.76% | +59.30% | +6.82% | -19.65% |
| 연승 | +14.48% | +0.18% | -0.12% | -24.87% |
| 복승 | +156.21% | +30.35% | +51.06% | +13.63% |
| 쌍승 | +148.28% | +26.32% | +69.88% | +27.79% |
| 삼복승 | +87.59% | +27.19% | -14.71% | -35.84% |
| 삼쌍승 | -100.00% | -100.00% | -100.00% | -100.00% |
| 복연승 | +25.17% | +4.74% | -5.53% | -28.94% |

## 단승·연승 안정성 상세

| bet_type | fraction_label | bets | wins | hit_rate | average_selected_odds | roi | roi_ci_low_daily_block | roi_ci_high_daily_block | flat_1pct_mdd | per_bet_sharpe | kelly_total_return | kelly_mdd | roi_without_largest_winning_return | odds_le_30_roi | positive_model_ev_bets | positive_model_ev_roi | roi_fdr_q_value | profit_confirmed_fdr_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 단승 | top_10pct | 29 | 5 | +17.24% | 19.717 | +112.76% | -37.67% | +309.64% | -14.85% | 1.205 | -0.11% | -0.11% | +48.93% | +146.80% | 1.0 | -100.00% | +89.72% | False |
| 연승 | top_10pct | 29 | 7 | +24.14% | 7.048 | +14.48% | -60.67% | +100.36% | -11.36% | 0.346 | +2.45% | -6.01% | -9.29% | +14.48% | 23.0 | -0.87% | +89.72% | False |
| 단승 | top_20pct | 57 | 7 | +12.28% | 22.614 | +59.30% | -45.28% | +174.31% | -15.71% | 0.985 | +3.22% | -0.11% | +26.43% | +97.39% | 4.0 | +627.50% | +89.72% | False |
| 연승 | top_20pct | 57 | 12 | +21.05% | 7.204 | +0.18% | -50.57% | +56.55% | -9.50% | 0.006 | +0.51% | -6.17% | -16.61% | +0.18% | 37.0 | -10.27% | +89.98% | False |
| 단승 | top_30pct | 85 | 7 | +8.24% | 27.669 | +6.82% | -62.28% | +87.96% | -22.88% | 0.166 | +3.04% | -0.14% | -15.71% | +53.90% | 8.0 | +263.75% | +89.72% | False |
| 연승 | top_30pct | 85 | 15 | +17.65% | 8.356 | -0.12% | -50.90% | +61.12% | -22.82% | -0.004 | +0.53% | -8.03% | -18.21% | +1.07% | 51.0 | -3.14% | +90.96% | False |
| 단승 | top_40pct | 113 | 7 | +6.19% | 28.579 | -19.65% | -71.32% | +41.01% | -36.92% | -0.631 | +3.04% | -0.14% | -36.79% | +17.92% | 9.0 | +223.33% | +100.00% | False |
| 연승 | top_40pct | 113 | 15 | +13.27% | 8.453 | -24.87% | -63.60% | +23.00% | -40.57% | -1.137 | -0.34% | -8.55% | -38.66% | -24.20% | 58.0 | -14.83% | +100.00% | False |

## test 상위 이변 말 예시

| model | race_id | hrName | market_rank | winOdds | model_place_probability | market_place_probability | predicted_edge | upset_score | win | place | top3_upset | top_positive_local_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 감사 후 전체 133피처 이변 Base Margin | 20260104_1_09 | 라온스필레타 | 7 | 10.4 | 29.255% | 19.935% | 9.320% | 0.093202 | 1 | 1 | 1 | tr_winrate__z:+0.0841 \| tool_n:+0.0443 \| hr_winrate__pr:+0.0396 \| tr_plcrate:+0.0285 \| hr_prev_rating:+0.0234 |
| 감사 후 전체 133피처 이변 Base Margin | 20260328_1_03 | 다이아비트 | 7 | 24.2 | 12.427% | 4.792% | 7.635% | 0.076348 | 0 | 0 | 0 | chulNo:+0.0419 \| hr_last_finpct:+0.0257 \| wg_diff__z:+0.0163 |
| 감사 후 전체 133피처 이변 Base Margin | 20260719_1_10 | 드래곤킹덤 | 6 | 10.6 | 31.905% | 24.292% | 7.613% | 0.076127 | 0 | 0 | 0 | tr_winrate__z:+0.0986 \| wgBudam:+0.0601 \| hr_winrate__z:+0.0417 \| rating__z:+0.0323 \| te_owName:+0.0303 |
| 감사 후 전체 133피처 이변 Base Margin | 20260419_1_08 | 스타트렉 | 6 | 10.3 | 28.763% | 21.547% | 7.216% | 0.072159 | 0 | 1 | 1 | te_owName:+0.0945 \| chulNo:+0.0653 \| wgBudam:+0.0637 \| wg_diff__z:+0.0412 \| hr_last_finpct:+0.0305 |
| 감사 후 전체 133피처 이변 Base Margin | 20260705_1_04 | 호라이즌스타 | 5 | 9.0 | 33.863% | 26.655% | 7.207% | 0.072071 | 0 | 0 | 0 | age__pr:+0.0505 \| hr_rest_days:+0.0402 \| hr_winrate__pr:+0.0323 \| tr_plcrate:+0.0171 \| age:+0.0146 |
| 감사 후 전체 133피처 이변 Base Margin | 20260621_1_10 | 남산미남 | 6 | 10.0 | 30.101% | 22.963% | 7.137% | 0.071371 | 0 | 0 | 0 | wgBudam:+0.1111 \| te_owName:+0.1027 \| wgBudam__pr:+0.0375 \| hr_last_finpct:+0.0267 \| chulNo:+0.0265 |
| 감사 후 전체 133피처 이변 Base Margin | 20260301_1_05 | 파워풀슬루 | 6 | 14.0 | 22.488% | 15.706% | 6.783% | 0.067828 | 0 | 0 | 0 | tr_winrate__z:+0.1193 \| tool_쿠션편자_Cushion:+0.0656 \| hr_last_finpct:+0.0612 \| hr_winrate__pr:+0.0339 \| tr_plcrate:+0.0305 |
| 감사 후 전체 133피처 이변 Base Margin | 20260620_1_06 | 에코스타 | 6 | 13.4 | 23.914% | 17.688% | 6.226% | 0.062262 | 0 | 0 | 0 | tr_winrate__z:+0.0696 \| age__pr:+0.0474 \| age:+0.0388 \| te_owName:+0.0286 \| te_trName:+0.0267 |
| 감사 후 전체 133피처 이변 Base Margin | 20260314_1_07 | 그레이스루트 | 7 | 20.0 | 17.085% | 11.036% | 6.049% | 0.060485 | 1 | 1 | 1 | wgBudam:+0.1164 \| te_owName:+0.0672 \| chulNo:+0.0653 \| style_vs_race:+0.0174 \| wgBudam__pr:+0.0155 |
| 감사 후 전체 133피처 이변 Base Margin | 20260725_1_02 | 마스크걸 | 9 | 38.2 | 10.183% | 4.183% | 6.000% | 0.060003 | 0 | 0 | 0 | tr_winrate__z:+0.1148 \| te_trName:+0.0423 \| tr_plcrate:+0.0285 \| tool_n:+0.0261 \| jk_winrate:+0.0156 |

위 말별 근거는 후보 피처를 같은 경주 중앙값으로 한 번씩 치환했을 때 시장 보정 margin이 얼마나 감소하는지를 계산한 국소 반사실 설명이다. 인과효과가 아니다.
