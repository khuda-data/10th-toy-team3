# 감사 후 전체 141피처 이변 말 탐색 및 수익성 보고서

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
- 피처 재학습 임계값은 **정규화된 양의 순열 중요도 0.5% 이상**인 운영 기준이다. 이는 p-value나 통계적 유의성 기준이 아니다.
- 선택 피처는 29개이며 목록은 `selected_upset_features.json`에 저장했다.
- 배당은 `ln(q)` 고정 offset과 평가에만 사용했고 일반 피처행렬에는 넣지 않았다. 최종 발매풀 5개(`winAmt·plcAmt·totalAmt·log_winAmt·liq_per_horse`)도 제외해 141피처를 사용했다.
- 복합 승식은 이변 말 한 마리를 반드시 포함하는 모든 가능한 티켓 중 모델 사건확률이 최대인 티켓을 택했다. 복합 시장확률은 q 기반 PL/Harville proxy이며 실제 풀 확률은 아니다.
- 재학습 타깃은 승자(`win`)이고, 연승확률은 경주별 승리확률에서 Plackett-Luce로 파생했다. `upset_B` 자체를 직접 분류한 모델은 아니다.

## 이변 피처 중요도 상위 30개

| rank | feature | positive_importance_weight | mean_upset_tail_realized_edge_drop | ci_low_daily_block | ci_high_daily_block | selected_for_retraining | ci_confirmed_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | clinic_30d | 16.152% | +0.026177 | -0.003293 | +0.058175 | True | False |
| 2 | tr_plcrate | 6.379% | +0.010338 | +0.003633 | +0.019392 | True | True |
| 3 | hr_last_finpct | 5.538% | +0.008974 | -0.001191 | +0.020357 | True | False |
| 4 | hr_rest_days | 4.885% | +0.007916 | +0.000194 | +0.017323 | True | True |
| 5 | age__pr | 4.583% | +0.007427 | -0.000522 | +0.017164 | True | False |
| 6 | hr_style | 4.354% | +0.007057 | -0.001582 | +0.018170 | True | False |
| 7 | oh_sex_암 | 4.165% | +0.006749 | -0.003749 | +0.019654 | True | False |
| 8 | wgBudam_chg | 3.973% | +0.006439 | +0.000414 | +0.013735 | True | True |
| 9 | ow_starts | 3.753% | +0.006082 | -0.001619 | +0.015881 | True | False |
| 10 | hr_plcrate | 3.703% | +0.006002 | +0.000188 | +0.012847 | True | True |
| 11 | jk_winrate | 3.400% | +0.005510 | -0.000825 | +0.015517 | True | False |
| 12 | waterRate | 3.267% | +0.005294 | -0.002167 | +0.017219 | True | False |
| 13 | hr_rest_days__z | 3.199% | +0.005184 | -0.004002 | +0.015254 | True | False |
| 14 | jk_winrate__z | 3.003% | +0.004867 | -0.006560 | +0.018826 | True | False |
| 15 | jk_plcrate | 2.922% | +0.004736 | -0.000717 | +0.011813 | True | False |
| 16 | wgBudam__pr | 2.871% | +0.004653 | -0.003476 | +0.014676 | True | False |
| 17 | hr_style_sd | 2.800% | +0.004537 | -0.002193 | +0.012941 | True | False |
| 18 | jk_starts | 2.516% | +0.004077 | -0.002400 | +0.010838 | True | False |
| 19 | ow_winrate | 2.311% | +0.003746 | -0.002314 | +0.010210 | True | False |
| 20 | train_sec_14 | 2.220% | +0.003598 | -0.003365 | +0.012269 | True | False |
| 21 | te_owName | 2.090% | +0.003387 | -0.005088 | +0.012587 | True | False |
| 22 | tr_winrate | 2.016% | +0.003267 | -0.004620 | +0.011952 | True | False |
| 23 | tool_망사눈가면 | 1.798% | +0.002914 | -0.001072 | +0.008245 | True | False |
| 24 | hr_winrate__pr | 1.731% | +0.002806 | -0.001921 | +0.007922 | True | False |
| 25 | age | 1.450% | +0.002350 | -0.000774 | +0.006021 | True | False |
| 26 | hr_dist_starts | 1.319% | +0.002137 | -0.001068 | +0.006348 | True | False |
| 27 | wg_diff | 1.062% | +0.001722 | -0.003062 | +0.005792 | True | False |
| 28 | hr_rest_days__pr | 1.004% | +0.001627 | -0.009109 | +0.013477 | True | False |
| 29 | tr_starts | 0.784% | +0.001271 | -0.006103 | +0.009621 | True | False |
| 30 | ill_n | 0.465% | +0.000754 | -0.004204 | +0.006906 | False | False |

## split별 정확 상위 비율과 점수 경계

| split | model_key | model | fraction | fraction_label | score_threshold | candidate_pool | selected_target_count | actual_selected_fraction | selection_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 10% | top_10pct | 0.05731479 | 574 | 58 | 10.10% | within-split score rank; outcomes unused |
| valid | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 20% | top_20pct | 0.03591585 | 574 | 115 | 20.03% | within-split score rank; outcomes unused |
| valid | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 30% | top_30pct | 0.02530918 | 574 | 173 | 30.14% | within-split score rank; outcomes unused |
| valid | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 40% | top_40pct | 0.02078010 | 574 | 230 | 40.07% | within-split score rank; outcomes unused |
| test | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 10% | top_10pct | 0.04496480 | 569 | 57 | 10.02% | within-split score rank; outcomes unused |
| test | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 20% | top_20pct | 0.03213525 | 569 | 114 | 20.04% | within-split score rank; outcomes unused |
| test | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 30% | top_30pct | 0.02392641 | 569 | 171 | 30.05% | within-split score rank; outcomes unused |
| test | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 40% | top_40pct | 0.01801042 | 569 | 228 | 40.07% | within-split score rank; outcomes unused |

## test 이변 말 구성과 실제 적중

| split | model_key | model | fraction | fraction_label | candidate_pool | selected_horses | actual_selected_fraction | mean_market_rank | mean_win_odds | actual_win_rate | actual_place_rate | stored_upset_B_rate | exact_top3_upset_rate | market_rank_2_3 | market_rank_4_6 | market_rank_7_plus | win_odds_lt_10 | win_odds_10_20 | win_odds_20_30 | win_odds_30_plus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 0.1 | top_10pct | 569 | 57 | 10.02% | 6.701754385964913 | 18.236842105263158 | 12.28% | 22.81% | 22.81% | 22.81% | 0 | 28 | 29 | 10 | 28 | 13 | 6 |
| test | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 0.2 | top_20pct | 569 | 114 | 20.04% | 7.043859649122807 | 22.68771929824561 | 7.02% | 21.05% | 21.05% | 21.05% | 0 | 49 | 65 | 11 | 50 | 28 | 25 |
| test | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 0.3 | top_30pct | 569 | 171 | 30.05% | 7.1988304093567255 | 25.02163742690059 | 5.26% | 21.05% | 21.05% | 21.05% | 0 | 68 | 103 | 15 | 65 | 49 | 42 |
| test | full_upset_base_margin | 감사 후 전체 피처 이변 Base Margin | 0.4 | top_40pct | 569 | 228 | 40.07% | 7.543859649122807 | 29.088157894736845 | 4.82% | 18.42% | 18.42% | 18.42% | 0 | 79 | 149 | 16 | 73 | 65 | 74 |

## 전체 피처 모델 test ΔLL

| model | races | delta_ll_per_race | delta_ll_ci_low_daily_block | delta_ll_ci_high_daily_block | delta_ll_confirmed_positive_95pct | model_top1_hit_rate | market_top1_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 감사 후 전체 피처 이변 Base Margin | 1068 | +0.012656 | +0.006053 | +0.019672 | True | 37.73% | 37.73% |

## 전체 피처 모델 test 승식별 ROI

| 승식 | top_10pct | top_20pct | top_30pct | top_40pct |
| --- | --- | --- | --- | --- |
| 단승 | +40.35% | -13.33% | -37.37% | -42.41% |
| 연승 | +14.56% | +17.46% | +15.85% | +5.39% |
| 복승 | -16.32% | +38.07% | +30.00% | +21.14% |
| 쌍승 | -31.58% | +79.91% | +74.50% | +53.99% |
| 삼복승 | +117.89% | +26.67% | +27.49% | +33.86% |
| 삼쌍승 | +269.30% | +84.65% | +220.47% | +296.18% |
| 복연승 | +11.05% | +0.44% | +5.56% | -3.29% |

## 단승·연승 안정성 상세

| bet_type | fraction_label | bets | wins | hit_rate | average_selected_odds | roi | roi_ci_low_daily_block | roi_ci_high_daily_block | flat_1pct_mdd | per_bet_sharpe | kelly_total_return | kelly_mdd | roi_without_largest_winning_return | odds_le_30_roi | positive_model_ev_bets | positive_model_ev_roi | roi_fdr_q_value | profit_confirmed_fdr_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 단승 | top_10pct | 57 | 7 | +12.28% | 18.237 | +40.35% | -49.11% | +149.67% | -12.25% | 0.759 | +4.42% | -1.08% | +7.14% | +56.86% | 20.0 | +192.00% | +54.34% | False |
| 연승 | top_10pct | 57 | 13 | +22.81% | 5.553 | +14.56% | -48.89% | +88.03% | -16.75% | 0.440 | +11.91% | -11.29% | -1.96% | +14.56% | 43.0 | +30.47% | +59.56% | False |
| 단승 | top_20pct | 114 | 8 | +7.02% | 22.688 | -13.33% | -67.31% | +51.50% | -27.59% | -0.423 | +6.28% | -1.35% | -30.27% | +11.01% | 33.0 | +133.94% | +82.81% | False |
| 연승 | top_20pct | 114 | 24 | +21.05% | 6.782 | +17.46% | -31.87% | +73.70% | -19.32% | 0.681 | +14.43% | -11.91% | +4.16% | +17.46% | 79.0 | +28.35% | +54.52% | False |
| 단승 | top_30pct | 171 | 9 | +5.26% | 25.022 | -37.37% | -74.54% | +5.35% | -53.00% | -1.722 | +5.37% | -1.57% | -48.76% | -16.98% | 41.0 | +88.29% | +99.16% | False |
| 연승 | top_30pct | 171 | 36 | +21.05% | 7.367 | +15.85% | -21.08% | +56.82% | -29.92% | 0.785 | +10.38% | -15.07% | +7.00% | +16.53% | 102.0 | +15.98% | +54.34% | False |
| 단승 | top_40pct | 228 | 11 | +4.82% | 29.088 | -42.41% | -72.78% | -6.64% | -64.87% | -2.367 | +4.74% | -2.04% | -50.97% | -14.74% | 53.0 | +45.66% | +100.00% | False |
| 연승 | top_40pct | 228 | 42 | +18.42% | 8.351 | +5.39% | -24.24% | +36.74% | -36.96% | 0.306 | +9.70% | -16.08% | -1.67% | +6.80% | 129.0 | +11.78% | +60.38% | False |

## test 상위 이변 말 예시

| model | race_id | hrName | market_rank | winOdds | model_place_probability | market_place_probability | predicted_edge | upset_score | win | place | top3_upset | top_positive_local_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 감사 후 전체 피처 이변 Base Margin | 20260515_3_06 | 탐라퍼스트 | 5 | 8.0 | 40.278% | 28.637% | 11.641% | 0.116407 | 0 | 0 | 0 | tr_winrate__z:+0.1361 \| age__pr:+0.0452 \| tr_plcrate:+0.0451 \| tr_starts:+0.0174 \| tr_winrate:+0.0123 |
| 감사 후 전체 피처 이변 Base Margin | 20260719_1_10 | 드래곤킹덤 | 6 | 10.6 | 33.926% | 24.292% | 9.634% | 0.096344 | 0 | 0 | 0 | tr_winrate__z:+0.1540 \| wgBudam:+0.1198 \| jkhr_starts:+0.0427 \| hr_winrate__z:+0.0421 \| hr_last_finpct:+0.0415 |
| 감사 후 전체 피처 이변 Base Margin | 20260612_3_01 | 태산신화 | 6 | 11.1 | 31.695% | 22.385% | 9.310% | 0.093102 | 0 | 0 | 0 | wg_diff:+0.0712 \| wgBudam:+0.0529 \| tr_winrate__z:+0.0411 \| hr_last_finpct:+0.0197 \| ow_plcrate:+0.0183 |
| 감사 후 전체 피처 이변 Base Margin | 20260104_1_09 | 라온스필레타 | 7 | 10.4 | 28.659% | 19.935% | 8.723% | 0.087235 | 1 | 1 | 1 | tr_winrate__z:+0.1099 \| tool_n:+0.0472 \| tr_plcrate:+0.0381 \| hr_winrate__pr:+0.0322 \| hr_prev_rating:+0.0233 |
| 감사 후 전체 피처 이변 Base Margin | 20260522_3_07 | 하늘만점 | 7 | 24.1 | 20.316% | 11.629% | 8.688% | 0.086876 | 0 | 0 | 0 | wgBudam:+0.1173 \| hr_dist_winrate:+0.0359 \| tr_starts:+0.0240 \| hr_rest_days__pr:+0.0221 \| hr_last_finpct:+0.0197 |
| 감사 후 전체 피처 이변 Base Margin | 20260109_3_08 | 셀라이크 | 8 | 11.5 | 25.509% | 17.279% | 8.230% | 0.082297 | 0 | 0 | 0 | tool_BlueBond편자:+0.1155 \| tool_쿠션편자_Cushion:+0.0408 \| hr_last_finpct:+0.0305 \| te_jkName:+0.0161 \| chulNo:+0.0107 |
| 감사 후 전체 피처 이변 Base Margin | 20251228_3_06 | 오늘이순간 | 7 | 11.6 | 29.968% | 22.182% | 7.786% | 0.077859 | 0 | 0 | 0 | clinic_30d:+0.2655 \| tr_winrate__z:+0.1577 \| tool_BlueBond편자:+0.0936 \| hr_winrate:+0.0361 \| hr_dist_starts:+0.0269 |
| 감사 후 전체 피처 이변 Base Margin | 20260215_3_02 | 타이거어퍼컷 | 6 | 12.7 | 33.398% | 25.636% | 7.762% | 0.077625 | 0 | 0 | 0 | tr_winrate__z:+0.1930 \| hr_rest_days__z:+0.1240 \| tool_n:+0.0619 \| ow_starts:+0.0376 \| wgBudam__pr:+0.0246 |
| 감사 후 전체 피처 이변 Base Margin | 20260614_3_06 | 윈드로즈 | 6 | 11.6 | 24.986% | 17.327% | 7.659% | 0.076590 | 0 | 0 | 0 | hr_last_finpct:+0.0197 \| wg_diff__pr:+0.0119 \| jk_plcrate:+0.0096 |
| 감사 후 전체 피처 이변 Base Margin | 20260302_1_07 | 서머파티 | 6 | 11.8 | 28.548% | 20.912% | 7.636% | 0.076359 | 1 | 1 | 1 | hr_rest_days__z:+0.0857 \| jkhr_starts:+0.0613 \| hr_rest_days__pr:+0.0344 \| tool_망사눈가면:+0.0250 \| hr_last_finpct:+0.0197 |

위 말별 근거는 후보 피처를 같은 경주 중앙값으로 한 번씩 치환했을 때 시장 보정 margin이 얼마나 감소하는지를 계산한 국소 반사실 설명이다. 인과효과가 아니다.
