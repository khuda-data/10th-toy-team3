# 베팅·EDGE·ROI 근거 인벤토리

- 작성 시점: 2026-08-23 KST
- 목적: 최종 종합 보고서가 수익성·시장 우위·EDGE에 관한 수치를 과장하지 않도록, 이미 생성된 CSV/JSON/로그만 재검산해 한곳에 고정한다.
- 기계 판독본: `outputs/reports/final_comprehensive_research_report_20260823/evidence/betting_edge_evidence.json`
- 현재 결론: **배치 가능한 수익 전략은 확인되지 않았다(NO_CONFIRMED_DEPLOYABLE_PROFIT_STRATEGY). 실베팅 권고액은 0이다.**
- 이 문서는 새 학습·재튜닝·백테스트·웹 검색·사후 정산을 수행하지 않았다. 기존 산출물 27개는 SHA-256까지 재검산했고, 누락 0개·해시 불일치 0개였다. 차트 후보 원천 19개도 모두 존재한다.

## 1. 판정 규칙과 용어

### 1.1 시장확률, EDGE, EV

한 경주 안에서 말 `i`의 단승 총배당을 `O_ri`라 할 때 정규화 시장확률은 다음과 같다.

`q_ri = (1 / O_ri) / Σ_j(1 / O_rj)`

따라서 경주별 `Σ_i q_ri = 1`이다. 모델확률을 `p_ri`라고 하면 단순 확률 EDGE는 다음과 같다.

`EDGE_ri = p_ri - q_ri`

양의 EDGE는 모델이 시장보다 높은 확률을 부여했다는 뜻일 뿐, 양의 수익기댓값을 보장하지 않는다. 1단위 티켓의 실행 가능한 총배당이 `O`일 때에만

`EV = p × O - 1`, `손익분기 확률 = 1 / O`

로 계산할 수 있다. 복합 승식은 **베팅 시점의 해당 조합 배당**이 없으면 진짜 EV와 Kelly를 식별할 수 없다.

실현 EDGE는 `I(사건 적중) - q`로 정의했다. fresh 정산의 EDGE 보정오차는 `평균(model EDGE) - 평균(realized EDGE)`이다.

### 1.2 DeltaLL, Brier, ROI, 위험

- `DeltaLL/race = mean_r[log p_model(winner_r) - log q_market(winner_r)]`. 양수면 모델의 우승마 로그손실이 시장보다 작다.
- `Brier improvement = Brier_market - Brier_model`. 양수면 모델이 유리하다.
- `ROI = (총환급 - 총베팅액) / 총베팅액`.
- `MDD`는 시간순 자금곡선의 고점 대비 최대 낙폭이다. 각 표에는 원본 열의 단위를 유지했다.
- 인용된 `sqrt(n) Sharpe`는 티켓별 평균수익을 표준편차로 나눈 뒤 `sqrt(베팅 수)`를 곱한 값이다. 별도 표기가 없으면 연율화 Sharpe가 아니다.
- Full Kelly는 `f* = (pO - 1)/(O - 1)`이고, fractional/capped Kelly는 `min(cap, max(0, c×f*))`이다. 연구용 계산일 뿐 실베팅은 없었다.
- 공식 최종배당은 공제율이 반영된 환급자료이므로 단승 20%, 다중승식 27%를 다시 차감하지 않았다. 이중 공제를 피했다.

### 1.3 불확실성과 다중검정

일별 블록 신뢰구간은 같은 날짜의 경주 의존성을 유지한다. 점추정 ROI가 양수라도 95% 구간이 0을 포함하면 수익 확인으로 판정하지 않는다. Holm은 고정된 검정군의 family-wise error를, Benjamini-Hochberg FDR은 기대 거짓발견 비율을 통제한다. 조정 후 통과하지 못한 셀은 전략으로 승격하지 않는다.

### 1.4 증거 범주

| 범주 | 이 문서에서의 의미 | 성과 확증 가능 여부 |
|---|---|---|
| historical/reused TEST | 적합 시점 기준 OOS일 수 있으나 이전 연구에서 이미 관찰한 구간 | 신규 독립 확증 아님 |
| retrospective rolling OOS | 시간순 rolling/expanding 적합은 지켰으나 날짜 전체가 과거 연구에 노출 | 강건성 근거, 독립 재현 아님 |
| later-date optional fresh64 | v11보다 뒤 날짜이나 initial27을 본 뒤 fresh37을 추가한 pooled64 | 보조 근거, pristine 확증 아님 |
| VALID selection only | VALID에서 정책을 잠갔고 TEST/target 결과는 사용하지 않음 | 가설 잠금일 뿐 성과 미확인 |
| prospective preregistered unsettled | 사전 고정된 프로토콜·모델·검증기이나 결과 미정산 | 현재 성과 주장 불가 |
| synthetic contract validation | 수식·평가기·게이트의 동작을 합성 데이터로 검사 | 수익성 근거 아님 |

## 2. 핵심 판정

1. v11 VALID 잠금 전체피처 정책과 시장 앵커 정책 모두 **확인된 수익 승식 0/7**이다.
2. later-date optional fresh64의 5피처 정산도 절대수익 확인 0/7, Holm 조정 시장우위 0/7이다. 유일한 양의 점 ROI는 쌍승 +0.94%지만 95% 구간은 -82.95%~+94.78%다.
3. 6피처 M6 fresh64도 양의 점 ROI가 한 승식도 없고 Holm 시장우위 0/7이다.
4. 가장 강한 방어적 신호는 과거 8~15배 연승 구간이다. fixed-lock에서 모델 ROI +1.30%, 동일경주 시장 대조 -30.50%, paired 차이 +31.80%이고 paired CI가 양수이며 Holm을 통과했다. 그러나 모델 절대 ROI CI는 -18.65%~+21.96%, TEST 재사용, 마감배당이므로 **수익 확증이 아니라 상대적 손실 방어**다.
5. v11 8~15배 재검증은 94건 ROI +9.79%이나 CI -46.31%~+74.21%다. 과거 +32.69%는 같은 크기로 재현되지 않았다.
6. 상위 10~40% upset grid는 56셀 중 점 ROI +30% 이상이 22셀이지만 CI 하한 양수 0/56, FDR 통과 0/56이다. 희귀 고배당 1~3회 적중에 집중된 기술적 결과다.
7. 확률 우위는 수익 우위보다 반복적으로 나타났다. 앵커 계열의 DeltaLL은 여러 구간에서 양수였지만 실행시점 배당, 표본 불확실성, 다중검정, 재사용 TEST 문제 때문에 실현 가능한 ROI로 전환되지 않았다.

## 3. v11 전체피처, 7개 승식

범위는 VALID에서 모델·정책을 선택하고 TEST(2025-12-28~2026-08-09)에 적용한 historical/reused TEST다. 원천은 `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/locked_policy_test_results.csv`이다.

| 승식 | 모델/정책 | 베팅/적중 | 적중률 | ROI [일별 95% CI] | 평균 선택배당 또는 평균 적중환급 | MDD | Sharpe | Kelly return | 시장 ROI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 단승 | RF / top10% EDGE | 108/18 | 16.67% | -3.06% [-51.28%, +53.23%] | 선택배당 8.880, 적중환급 5.817 | -21.05% | -0.109 | -19.89% | -17.60% |
| 연승 | XGB Ranker / top10% | 103/47 | 45.63% | +1.26% [-26.41%, +29.55%] | 선택배당 2.696, 적중환급 2.219 | -15.21% | +0.098 | -0.78% | -15.34% |
| 복승 | RF / positive EDGE | 685/63 | 9.20% | -17.65% [-41.32%, +9.54%] | 적중환급 8.954 | -83.82% | -1.256 | 식별 불가 | -8.46% |
| 쌍승 | RF / positive EDGE | 688/35 | 5.09% | -18.04% [-51.42%, +25.37%] | 적중환급 16.111 | -90.38% | -0.930 | 식별 불가 | -13.90% |
| 삼복승 | XGB Ranker / top30% | 324/15 | 4.63% | -24.54% [-65.81%, +26.54%] | 적중환급 16.300 | -71.13% | -1.069 | 식별 불가 | -24.60% |
| 삼쌍승 | RF / positive EDGE | 752/4 | 0.53% | -66.52% [-95.66%, -21.89%] | 적중환급 62.950 | -99.67% | -3.330 | 식별 불가 | -33.49% |
| 복연승 | LGB LambdaRank / top10% | 108/10 | 9.26% | -53.61% [-81.22%, -20.80%] | 적중환급 5.010 | -47.89% | -3.593 | 식별 불가 | -20.74% |

연승만 점 ROI가 양수이나 CI가 0을 포함한다. 삼쌍승과 복연승은 CI 전체가 0 아래다. 따라서 전체피처 7개 승식에서 수익 확정은 0개다.

## 4. v11 시장 앵커

확률 원천은 `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/delta_ll_metrics.csv`, 베팅 원천은 같은 폴더의 `locked_test_results.csv`다. 1,101개 reused TEST 경주에서:

| 확률모델 | DeltaLL/race [95% CI] |
|---|---:|
| anchored conditional logit | +0.013974 [+0.002318, +0.025063] |
| anchored Base Margin | +0.013083 [+0.006614, +0.019647] |
| anchor ensemble | +0.017194 [+0.009507, +0.025014] |

세 확률모델은 시장보다 로그손실이 낮았지만, 7개 ROI는 모두 불확실했다.

| 승식 | 모델/정책 | 베팅/적중 | ROI [95% CI] | 평균 EDGE | EDGE-실현 Spearman | 평균 적중환급 | MDD | Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 단승 | Base Margin / top10% | 108/45 | -14.54% [-35.54%, +6.77%] | 0.0709 | -0.0412 | 2.051 | -17.23% | -1.396 |
| 연승 | Base Margin / top10% | 101/74 | -7.52% [-17.01%, +2.38%] | 0.2734 | +0.0989 | 1.262 | -8.86% | -1.316 |
| 복승 | ensemble / top10% | 108/34 | +39.72% [-3.30%, +84.45%] | 0.0649 | -0.1615 | 4.438 | -12.63% | +1.785 |
| 쌍승 | ensemble / top10% | 108/21 | +16.76% [-31.05%, +67.62%] | 0.0429 | -0.1155 | 6.005 | -21.35% | +0.643 |
| 삼복승 | Base Margin / top20% | 216/34 | -7.69% [-39.63%, +26.52%] | 0.0438 | -0.0616 | 5.865 | -39.33% | -0.463 |
| 삼쌍승 | Base Margin / top10% | 108/5 | -32.96% [-86.74%, +32.35%] | 0.0175 | -0.2895 | 14.480 | -43.27% | -1.014 |
| 복연승 | ensemble / top10% | 108/52 | +15.93% [-5.85%, +39.13%] | 0.1005 | +0.0677 | 2.408 | -12.12% | +1.261 |

복승의 +39.72%가 가장 크지만 CI 하한 -3.30%다. **확률 개선과 현금흐름 확증은 별개**라는 프로젝트의 핵심 반증 사례다.

## 5. 이변 피처 중요도와 상위 10~40%

원천:

- `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/upset_feature_importance.csv`
- `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/model_delta_ll_metrics.csv`
- `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/upset_bet_summary_independently_verified.csv`

중요도는 이변 꼬리의 실현 EDGE 목적에 대해 경주 내 또는 동일 출전두수 race-block permutation으로 계산하고, 일별 블록 불확실성을 붙였다. 0.5%(`weight > 0.005`)는 **운영용 선택 문턱**이며 유의수준이 아니다. 141개 전체피처 모델 DeltaLL은 +0.012656 [0.006053, 0.019672], 29개 선택피처 모델은 +0.010771 [0.004139, 0.017539]였다.

선택된 29개 순위는 다음과 같다.

| 순위 | 피처 | 정규화 중요도 | 순위 | 피처 | 정규화 중요도 |
|---:|---|---:|---:|---|---:|
| 1 | clinic_30d | 0.161525 | 16 | wgBudam__pr | 0.028710 |
| 2 | tr_plcrate | 0.063790 | 17 | hr_style_sd | 0.027998 |
| 3 | hr_last_finpct | 0.055378 | 18 | jk_starts | 0.025158 |
| 4 | hr_rest_days | 0.048847 | 19 | ow_winrate | 0.023113 |
| 5 | age__pr | 0.045826 | 20 | train_sec_14 | 0.022202 |
| 6 | hr_style | 0.043543 | 21 | te_owName | 0.020898 |
| 7 | oh_sex_암 | 0.041646 | 22 | tr_winrate | 0.020160 |
| 8 | wgBudam_chg | 0.039733 | 23 | tool_망사눈가면 | 0.017984 |
| 9 | ow_starts | 0.037530 | 24 | hr_winrate__pr | 0.017312 |
| 10 | hr_plcrate | 0.037033 | 25 | age | 0.014503 |
| 11 | jk_winrate | 0.033998 | 26 | hr_dist_starts | 0.013189 |
| 12 | waterRate | 0.032666 | 27 | wg_diff | 0.010624 |
| 13 | hr_rest_days__z | 0.031987 | 28 | hr_rest_days__pr | 0.010040 |
| 14 | jk_winrate__z | 0.030035 | 29 | tr_starts | 0.007843 |
| 15 | jk_plcrate | 0.029224 |  |  |  |

중요도 CI가 온전히 양수인 피처는 네 개뿐이다.

| 순위 | 피처 | weight | 일별 95% CI |
|---:|---|---:|---:|
| 2 | tr_plcrate | 0.063790 | [+0.003633, +0.019392] |
| 4 | hr_rest_days | 0.048847 | [+0.000194, +0.017323] |
| 8 | wgBudam_chg | 0.039733 | [+0.000414, +0.013735] |
| 10 | hr_plcrate | 0.037033 | [+0.000188, +0.012847] |

`clinic_30d`는 점중요도 1위지만 H7 ablation의 증분은 확인되지 않았다. 중요도와 독립적인 증분 효용을 구분해야 한다.

상위 10/20/30/40% × 2모델 × 7승식 = 56개 ROI 셀 중:

- 점 ROI +30% 이상: 22/56
- 조정 전 CI 하한 양수: 0/56
- 5% FDR 통과: 0/56

대표적인 큰 점추정치는 full/top40/삼쌍승 228건·3적중 ROI +296.18% [−100%, +782.35%], 최대 적중 제거 후 +141.41%; full/top10/삼쌍승 57건·1적중 +269.30%, 최대 적중 제거 후 −100%; selected/top40/삼쌍승 232건·3적중 +240.65%, 최대 적중 제거 후 +88.31%; selected/top20/쌍승 116건·7적중 +110.34%, 최대 적중 제거 후 +50.00%다. full/top10/단승은 57건·7적중 +40.35% [−49.11%, +149.67%], 최대 적중 제거 후 +7.14%, Kelly return +4.42%, Kelly MDD −1.08%다. 어느 셀도 확증 기준을 넘지 못했다.

## 6. 8~15배 다크호스 가설

### 6.1 v11 재검증

`outputs/reports/odds_8_15_v11_revalidation_20260823/summary.json`에서 Base Margin의 VALID 잠금 EDGE 문턱은 `0.0188698053214846`이다. 마감 단승배당 `[8, 15)`인 TEST 94건에서 적중률 11.70%, 평균배당 9.779, ROI +9.79%, CI −46.31%~+74.21%다. 동일 수 대조용 시장-q ROI는 −12.45%, 과거 batch top5 ROI는 +1.18%다. 수익 확인은 아니다.

### 6.2 H11C VALID 잠금

원천은 `outputs/reports/h11c_darkhorse_8_15_preregistered_20260823/valid_cutoff_metrics.csv`와 `valid_paired_metrics.csv`다. 2,022행·971경주의 VALID 8~15 구간에서 H11B Brier 0.06799975, 시장 0.06805533으로 개선량은 0.00005558에 불과했다. ECE는 H11B 0.005920, 시장 0.000847로 시장이 더 잘 보정됐다.

| cutoff | 잠금 EDGE 문턱 | H11B 티켓 | H11B ROI [CI] | 동일 수 q 대조 ROI | paired 차이 | Holm |
|---|---:|---:|---:|---:|---:|---|
| top10 | 0.00971384 | 87 | −10.46% [−65.06%, +56.97%] | +10.46% | −20.92% | 실패 |
| top20 | 0.00704490 | 173 | −8.96% [−48.91%, +37.06%] | +3.41% | −12.37% | 실패 |
| top30 | 0.00544431 | 259 | −12.93% [−45.27%, +21.88%] | +9.50% | −22.43% | 실패 |
| top40 | 0.00417516 | 345 | −3.36% [−31.84%, +27.68%] | +1.54% | −4.90% | 실패 |

VALID 단계 자체에서 H11B는 q 대조를 이기지 못했다. 따라서 8~15배는 보존할 만한 연구 가설이지만 자동 실전 규칙은 아니다.

## 7. H5~H9: 방어적 신호, 보정 실패, later-date optional 정산

### H5 adaptive rolling

`outputs/reports/rolling_origin_market_challenger_20260823/experiment_summary.json`: 실제 적합 32회, rolling OOS 2,185경주. DeltaLL +0.009587 [0.002457, 0.016667], Holm 통과. 8~15배 연승 366건의 모델 ROI +2.43% [−13.41%, +18.80%], 동일경주 시장 −21.01%, paired +23.44% [6.20%, 41.77%], Holm 통과. 적중률 37.16%, 평균배당 2.866, 평균 적중환급 2.757, MDD −17.74%, Sharpe +0.338, Kelly return +33.57%, Kelly MDD −28.30%다. 시간순은 지켰지만 retrospective/reused dates다.

### H5 fixed-lock

`outputs/reports/fixed_lock_rolling_stability_20260823/experiment_summary.json`: 실제 적합 32회, OOS 1,101경주, 고정 EDGE 문턱 0.07096038. DeltaLL +0.012046 [0.003180, 0.021159], Holm 통과. 연승 200건 모델 ROI +1.30% [−18.65%, +21.96%], 시장 −30.50%, paired +31.80% [10.51%, 53.76%], Holm 통과. 적중률 37.50%, 평균배당 2.869, 평균 적중환급 2.701, MDD −15.83%, Sharpe +0.136, Kelly return −0.81%, Kelly MDD −22.79%다. 이 문서에서 가장 강한 **상대적 손실 방어**지만 절대수익은 미확인이다.

### H6 확률보정

`outputs/reports/rolling_place_probability_calibration_20260823/experiment_summary.json`: 보정기 4개, 961관측, 선택 200건. 보정 후 uncalibrated 대비 LL +0.005231 [0.000118, 0.010480]이나 Holm 실패, 시장 대비 +0.001891 [−0.005565, 0.009349], Holm endpoints 0개다. Kelly return은 보정 전 −0.81%에서 보정 후 −18.83%, Kelly MDD −19.30%로 악화했다. proper score 개선이 수익 개선을 보장하지 않았다.

### H7 clinic ablation

`outputs/reports/clinic30d_rolling_ablation_20260823/experiment_summary.json`: 적합 40회, ablation 32회, TEST 1,101경주. global increment +0.001300 [−0.002530, 0.005115], partial8 +0.000943 [−0.003414, 0.005404], Holm 통과 0개. `clinic_30d`의 높은 permutation 중요도는 독립 증분 확인이 아니다.

### H8 later-date fresh64 확률

`outputs/reports/fresh_minimal_entry_market_challenger_extended_20260823/experiment_summary.json` 및 `outputs/reports/incremental_fresh_holdout_20260823/experiment_summary.json`: 5피처 실제 학습, 64경주·6일. 시장 대비 DeltaLL +0.018306, 일별 CI [0.001842, 0.040309], 경주별 CI [−0.001997, 0.038029], Brier 개선 +0.000746. 추가 fresh37만의 DeltaLL +0.028644, 일별 CI [0.002391, 0.056356]. 날짜는 뒤지만 initial27 관찰 후 추가된 optional continuation이다.

### H9 fresh64 공식 7승식

원천은 `outputs/reports/fresh_all_bets_extended_20260823/fresh_all_bet_summary.csv`와 `outputs/reports/fresh_all_bets_paired_advantage_extended_20260823/paired_advantage_summary.csv`다. 64경주·6일, 공식 페이지 7개, 후보 티켓 69,276개를 검사했다. 절대수익 확정 0/7, Holm paired 시장우위 0/7, 모든 복합 승식의 실제 EV/Kelly 식별 불가다.

| 승식 | 베팅/적중 | 적중률 | ROI [일별 95% CI] | 평균 적중환급 | MDD | Sharpe | 평균 EDGE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 단승 | 64/18 | 28.13% | −17.81% [−61.82%, +43.03%] | 2.922 | −16.89% | −0.967 | 0.02180 |
| 연승 | 64/33 | 51.56% | −12.34% [−34.92%, +5.93%] | 1.700 | −8.90% | −1.072 | 0.03506 |
| 복승 | 64/8 | 12.50% | −17.81% [−66.78%, +28.31%] | 6.575 | −16.34% | −0.604 | 0.01791 |
| 쌍승 | 64/6 | 9.38% | +0.94% [−82.95%, +94.78%] | 10.767 | −18.61% | +0.021 | 0.01114 |
| 삼복승 | 64/6 | 9.38% | −17.81% [−76.43%, +48.38%] | 8.767 | −21.90% | −0.502 | 0.01666 |
| 삼쌍승 | 64/0 | 0.00% | −100% [−100%, −100%] | 없음 | −64.00% | 정의 불가 | 0.00467 |
| 복연승 | 64/17 | 26.56% | −13.44% [−54.81%, +35.86%] | 3.259 | −11.80% | −0.653 | 0.03454 |

## 8. H10~H18

### H10A: 재현 가능한 6피처 앵커와 fresh64 반증

`outputs/reports/h10a_provenance_six_feature_anchor_20260823/endpoint_metrics.csv`: `age`, `age__pr`, `oh_sex_암`, `wgBudam_chg`, `wgBudam__pr`, `te_owName` 여섯 피처로 실제 적합 6회. X에 시장·결과 피처는 없었다. VALID M6-q DeltaLL +0.001585 [−0.003882, 0.006942], reused TEST +0.005422 [0.001236, 0.009480]와 Holm 통과, fresh64 +0.010457 [−0.001784, 0.024654], exact sign-flip p=0.078125다. fresh64에서 M6−M5는 −0.007904여서 배포 판정은 NO_GO다.

`outputs/reports/h10a_provenance_six_feature_all_bets_posthoc_20260823/fresh_all_bet_summary.csv`의 64경주 7승식 ROI는 단승 −17.66%, 연승 −27.66%, 복승 −22.50%, 쌍승 −35.16%, 삼복승 −35.94%, 삼쌍승 −100%, 복연승 −12.19%다. 양의 점 ROI 0/7, Holm paired 우위 0/7이다.

### H11B: 실제 다중모델 학습과 prospective freeze

`outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/experiment_summary.json`: 실제 fit operation 33회, 성공 최종 fit 15개, 중단되었지만 보존된 fit operation 18개. 구성원은 M6, XGBoost, LightGBM, deep MLP 3-seed, equal ensemble이다. 경주별 확률합 최대오차 `4.44e-16`, 재로드 예측 최대오차 0이다.

VALID DeltaLL은 M6 +0.001585, XGB +0.000654, LGB +0.002629, deep +0.000641, ensemble +0.002420이며 모두 CI가 0을 포함했다. reused TEST는 M6 +0.005422, XGB −0.001493, LGB −0.002300, deep +0.005182, ensemble +0.002722다. 훈련 종료 당시 target-date 점수 생성 없음, target outcome read 없음, stake 0이었다.

### H11D/H11E

- H11D `outputs/reports/h11d_market_anchor_shrinkage_preregistered_20260823/experiment_summary.json`: development alpha 1.05, late28 340경주 DeltaLL +0.002289 [−0.003839, 0.008967], p=0.49775; all-VALID 배포 alpha 1.06, stake 0.
- H11E `outputs/reports/h11e_prior_best_deployment_feasibility_20260823/feasibility_audit.json`: 과거 29피처 중 provenance-ready 6개, 차단 23개. clinic/training/equipment/track와 시간안전 이력을 정확히 재구성할 수 없어 대리값·중앙값 추측을 거부했고, target score를 만들지 않았다. 상태는 `BLOCKED_NO_EXACT_FEATURE_REPLAY`다.

### H13/H14/H17: 앵커 풀링과 규제

| 실험 | 실제 optimizer fit | 구조/선택 | late28 DeltaLL [95% CI] | 안정성/해석 |
|---|---:|---|---:|---|
| H13 nonnegative log pool | 8 | lambda 0; first64 XGB .2382/LGB .5847/deep .3224 | +0.002006 [−0.004539, +0.009043] | first64→all92 L1 1.218, 불안정 |
| H14 convex residual pool | 3 | residual 합≤1; first64 합 1, market exponent 0 | +0.002263 [−0.003475, +0.008439] | L1 0.913, 불안정 |
| H17 ridge convex pool | 14 | lambda 1; residual 합 .01923, market exponent .98077 | +0.0000837 [−0.0000295, +0.0002075] | 대부분 시장으로 수축, p=.0943 |

원천은 각각 `outputs/reports/h13_market_anchor_nonnegative_log_pool_preregistered_20260823/experiment_summary.json`, `outputs/reports/h14_convex_market_residual_pool_preregistered_20260823/experiment_summary.json`, `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/experiment_summary.json`이다. 세 실험 모두 ROI 미계산, stake 0이다. 규제가 신호를 거의 시장으로 되돌린 것은 raw ensemble 개선이 추정 잡음에 비해 약했을 가능성을 지지한다.

### H12/H15/H16/H18: 평가기 준비도이지 성과가 아님

- H12 ticket validator: 38 pass, 0 fail, 결과 미열람, stake 0.
- H15 probability evaluator: 132 pass, 0 fail, `GATE_CLOSED_NO_TARGET_RESULT_READ`, stake 0.
- H16 ticket extension: 1,514 pass, 0 fail, target outcome/dividend read 0, stake 0.
- H18 ticket extension: 128 pass, 0 fail, target outcome/dividend read 0, stake 0.
- 결합 settlement evaluator: freeze 4,229 pass, cross-component 208 pass, metric recalculation 2,820 pass, negative contract 35 pass, 모두 0 fail. target outcome row 0, network call 0, one-day profit false, stake 0.

결합 원천은 `outputs/reports/h16_h18_settlement_cross_component_audit_20260823/cross_component_audit_v2.json`이다. 이 PASS들은 공식·수식·게이트 구현 준비만 증명하며 ROI·EV·적중률·배당·시장우위를 증명하지 않는다.

## 9. 2026-08-23 prospective 상태: 미정산

인벤토리 고정시각은 `2026-08-23T10:19:36+09:00`이다. 예정 17경주 모두 H11/H11B 상태 `WAITING_T5_WIN_PLACE`; H13/H14/H16/H17/H18 frozen race 0, target outcome/result/dividend row read 0, network settlement call 0, live stake 0이다. 따라서 당일 성과 주장을 만들 수 없다. watcher state는 예약 작업이 움직이면 바뀌는 mutable 파일이다.

상태 경로:

- `outputs/reports/h11_h11b_prospective_t5_scoring_20260823/watcher_state.json`
- `outputs/reports/h13_prospective_t5_scoring_20260823/watcher_state.json`
- `outputs/reports/h14_prospective_t5_scoring_20260823/watcher_state.json`
- `outputs/reports/h16_prospective_all_model_seven_bet_extension_preregistered_20260823/ticket_watcher_state.json`
- `outputs/reports/h17_prospective_t5_scoring_20260823/watcher_state.json`
- `outputs/reports/h18_h17_prospective_probability_and_seven_bet_extension_preregistered_20260823/ticket_watcher_state.json`

## 10. Deep/stack 및 slippage 방어 결과

`outputs/reports/deep_market_challengers_20260823_v2/experiment_summary.json`: 실제 학습, parameter file 12개. 최선 reused TEST DeltaLL +0.010400 [0.002282, 0.018508], profit FDR cell 0, market-advantage FDR cell 0. `outputs/reports/market_residual_log_pool_stack_20260823/experiment_summary.json`의 log pool도 실제 학습했고 DeltaLL +0.009772 [0.001578, 0.017560], 수익/시장우위 FDR cell 모두 0이다.

deep blend의 reused TEST 8~15배 연승은 82건, 모델 ROI −0.37% [−32.06%, +30.86%], 동일경주 시장 −37.68%, paired +37.32% [4.62%, 71.53%]다. 그러나 paired FDR q=0.09567로 5%를 통과하지 못했다. 절대수익 미확인, reused TEST, 마감배당이라는 한계가 있다.

`outputs/reports/deep_market_challengers_20260823_v2/independent_audit/payout_slippage_sensitivity.csv`의 payout haircut 민감도:

| haircut | 모델 ROI |
|---:|---:|
| 0% | −0.37% |
| 2% | −2.36% |
| 5% | −5.35% |
| 10% | −10.33% |
| 20% | −20.29% |

근소한 성과는 가격악화에 취약하다. 이 결과는 수익 전략보다 손실방어 세그먼트를 탐색할 근거다.

## 11. 과거·대체·비확증 결과

| 실험 | 확률 결과 | 베팅 결과 | 판정/원천 |
|---|---|---|---|
| Benter anchored walk-forward | 4,009경주 DeltaLL +0.004906 [0.001256, 0.008635] | 327건 ROI +32.69% [−2.70%, +68.25%] | 미확증, v11에서 같은 크기로 미재현; `outputs/reports/benter_market_anchored_20260820/` |
| Base Margin boosting | 4,009경주 DeltaLL +0.007392 [0.004373, 0.010444] | 579건 ROI +2.57% [−31.24%, +41.00%] | 미확증; `outputs/reports/base_margin_boosting_20260821/summary.json` |
| two-step crossfit GBM | DeltaLL +0.002199 [0.000436, 0.003964] | 475건 ROI −27.37% [−58.29%, +7.40%] | 수익 실패; `outputs/reports/benter_two_step_20260821/summary.json` |
| historical 8~15 top5% | - | ROI +47.47% [14.16%, 82.73%] | 동일 역사 데이터에서 정책 선택했다고 원천이 명시; OOS 수익 추정 아님; `outputs/reports/base_margin_profit_20260821/summary.json` |
| PBO audit | PBO .00794; deflated-Sharpe probability .04815 | expected max Sharpe .24748 > observed .15241 | 전체 연구 자유도의 일부만 포함, 선택편향 하한; `outputs/reports/backtest_overfit_20260821/summary.json` |
| old v7 137/61 | - | 상위 EDGE fraction 결과 | 서울 전용이며 137 경로에 final-pool 누수 발견; v11이 대체; `outputs/reports/edge_fraction_10_30_retrain_20260820/top_edge_10_30/top_edge_fraction_roi_manifest.json` |

## 12. 수수료·가격·Kelly의 식별 경계

`outputs/reports/h11_multibet_realtime_odds_source_audit_20260823/source_audit.json`을 기준으로 안전한 공식 T-5 full-grid 7승식 가격은 검증되지 않았다. 연승·쌍승·복연승·삼복승·삼쌍승의 안전한 사전 조합배당 grid가 없고, 기존에 관찰된 단승·복승 public 경로도 결과를 함께 반환할 수 있는 contract라 사전정보 안전성이 해결되지 않았다.

따라서:

- 마감/결과 단승배당은 T-5 실행가격으로 문서화되지 않았다.
- 적중 조합의 최종환급만으로 모든 손실 티켓의 사전 EV를 역산할 수 없다.
- 복합 승식 Kelly를 q-derived proxy odds로 계산해도 실행 가능한 EV 근거가 되지 않는다.
- 공식 최종환급에는 공제율이 이미 반영됐으므로 20%/27% 재차감은 하지 않는다.
- near-zero ROI는 2~20% payout haircut에서 빠르게 악화했다.

## 13. 실패·한계 체크리스트

1. 적용 가능한 다중검정 이후, VALID-locked 또는 later-date 절대 ROI의 95% 하한이 0보다 큰 승식은 없다.
2. fresh64는 6개 경주일뿐이다. exact sign-flip p-value 해상도도 1/64이고 ROI 구간이 매우 넓다.
3. fresh37은 initial27을 본 후 수집되었으므로 pooled64는 optional continuation이다.
4. 복합승식의 큰 점 ROI는 극소수 적중에 집중되고 최대 적중 제거 시 크게 무너진다.
5. H11C의 8~15배 H11B cutoffs는 VALID에서조차 동일 수 q 대조를 이기지 못했고 Holm 0/4였다.
6. 보정은 proper score를 개선했지만 Kelly return을 −0.81%에서 −18.83%로 악화시켰다.
7. clinic_30d 중요도 1위와 달리 ablation CI는 0을 포함했다.
8. H13/H14 풀링 weight는 시기별로 불안정했다. H17 규제는 안정화 대신 신호를 거의 시장으로 수축시켰다.
9. 역사적으로 가장 강했던 29피처를 prospective하게 재현하려면 23개 피처의 정확한 원천이 부족했고, 프로젝트는 추측을 거부했다.
10. 한 target 경주일 정산만으로 장기 ROI·MDD·Sharpe·시장우위를 확증할 수 없다.
11. 현재 prospective trial은 미정산이며 성과관측 0개다.

## 14. 차트에 바로 사용할 기존 데이터셋

새 수치를 만들지 않고 다음 원천을 그대로 시각화할 수 있다.

| 목적 | 데이터 경로 | 권장 차트 |
|---|---|---|
| 전체피처 7승식 | `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/locked_policy_test_results.csv` | 승식별 ROI·CI, 시장 ROI overlay |
| 시장 앵커 7승식 | `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/locked_test_results.csv` | ROI/MDD 비교 |
| 앵커 확률우위 | `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/delta_ll_metrics.csv` | DeltaLL forest plot |
| 이변 피처 | `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/upset_feature_importance.csv` | 중요도+CI+0.5%선 |
| top10~40 | `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/upset_bet_summary_independently_verified.csv` | 모델×비율×승식 ROI heatmap, 최대적중 제거 병기 |
| 8~15 v11 | `outputs/reports/odds_8_15_v11_revalidation_20260823/odds_8_15_policy_results.csv` | 정책/모델별 ROI forest |
| H5 adaptive | `outputs/reports/rolling_origin_market_challenger_20260823/fold_place_policy_results.csv` | fold별 모델/시장/paired ROI |
| H5 fixed | `outputs/reports/fixed_lock_rolling_stability_20260823/fold_place_metrics.csv` | fold 안정성·hit-rate delta |
| H6 보정 | `outputs/reports/rolling_place_probability_calibration_20260823/calibration_metrics.csv` | calibrated/uncalibrated/q proper score |
| H9 fresh64 | `outputs/reports/fresh_all_bets_extended_20260823/fresh_all_bet_summary.csv` | 7승식 ROI·CI·MDD·Sharpe·적중률 dashboard |
| H9 paired | `outputs/reports/fresh_all_bets_paired_advantage_extended_20260823/paired_advantage_summary.csv` | 모델−시장 paired forest |
| H10A 확률 | `outputs/reports/h10a_provenance_six_feature_anchor_20260823/endpoint_metrics.csv` | VALID/reused TEST/fresh64 DeltaLL |
| H10A 7승식 | `outputs/reports/h10a_provenance_six_feature_all_bets_posthoc_20260823/fresh_all_bet_summary.csv` | 5피처와 join한 7승식 ROI |
| H11B 모델군 | `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/historical_probability_metrics.csv` | 모델별 VALID/TEST DeltaLL·Brier |
| H11C 다크호스 | `outputs/reports/h11c_darkhorse_8_15_preregistered_20260823/valid_cutoff_metrics.csv` | cutoff별 H11B/q ROI·MDD |
| H13 | `outputs/reports/h13_market_anchor_nonnegative_log_pool_preregistered_20260823/late28_metrics.csv` | q/ensemble proper scores |
| H14 | `outputs/reports/h14_convex_market_residual_pool_preregistered_20260823/late28_metrics.csv` | H14/H13/equal/q 비교 |
| H17 | `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/late28_metrics.csv` | 규제 수축과 DeltaLL |
| slippage | `outputs/reports/deep_market_challengers_20260823_v2/independent_audit/payout_slippage_sensitivity.csv` | haircut-ROI 곡선 |

## 15. 보고서에 사용할 수 있는 정직한 최종 문장

> 현재 결과는 모델이 시장확률보다 더 나은 확률 예측을 하는 구간이 있음을 반복적으로 보여주지만, 한국마사회 최종환급과 현재의 가격·표본 제약을 반영했을 때 7개 승식 중 통계적으로 확정된 절대수익 전략은 없다. 재사용 OOS 8~15배 연승 구간에서는 동일경주 시장 대조보다 손실을 줄이는 paired 우위가 가장 강하게 나타났으나, 절대 ROI는 0과 구분되지 않았고 마감배당·선택편향·다중검정 한계가 남는다. 따라서 현 단계의 결론은 ‘수익 전략 확보’가 아니라 ‘시장 앵커를 바탕으로 한 제한적 확률우위와 특정 구간의 방어 가능성’이며, 실베팅은 하지 않고 완전 신규 forward OOS와 사전 실행가능 조합배당을 확보한 뒤 재확인해야 한다.

## 16. 무결성 메모

기계 판독 JSON의 `artifact_manifest`에는 핵심 원천 27개의 상대경로, 바이트 수, SHA-256이 들어 있다. 현 작업에서 27/27 경로 존재, 27/27 해시 일치, 차트 원천 19/19 존재를 확인했다. 수치는 해당 파일에서 직접 읽었고, 미정산 watcher는 고정시각만 기록해 향후 변동과 분리했다.
