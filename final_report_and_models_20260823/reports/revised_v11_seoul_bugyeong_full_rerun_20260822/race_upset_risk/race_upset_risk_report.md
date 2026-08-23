# v11 서울+부경 경주 단위 이변 위험·관망 계층 보고서

> **증거 등급 주의:** 현재 test 기간은 프로젝트의 이전 분석에서 이미 관찰됐다. 이 실행은 test로
> 모델·임계값·정책을 선택하지 않았지만, 결과 자체는 **탐색적 재사용**이며 신규 미관측 기간의
> 전향적 확증으로 간주할 수 없다.

## 0. 지표와 용어

- **이변 레이블(upset label):** Within each race, rank final winOdds ascending (rank 1 = lowest odds) with average ranks for equal odds.  Let c=ceil(field_size/2).  upset_label=1 iff the winner's average market rank is strictly greater than c.  A tied odds group crossing c is treated together according to its average rank, so the realized bottom group need not contain exactly floor(N/2) runners.
- **ROC-AUC:** 무작위 이변 경주와 비이변 경주 한 쌍에서 이변 경주의 위험점수를 더 높게 줄 확률이다.
- **PR-AUC:** 드문 이변(양성)에 초점을 둔 정밀도-재현율 곡선의 면적이다. 양성률과 함께 읽어야 한다.
- **Brier:** `(예측확률-실제값)^2`의 평균이며 0에 가까울수록 좋다.
- **ECE(10-bin):** 고정 10개 확률구간에서 예측확률과 관측률 차이의 표본가중 평균이다. 0이 이상적이다.
- **Calibration intercept/slope:** 예측 logit을 다시 로짓 회귀한 절편/기울기다. 이상값은 각각 0/1이다.
- **Observed upset rate:** 실제 이변 레이블의 평균이다.
- **Lift:** 선택된 위험구간의 이변률을 같은 split·경마장 전체 이변률로 나눈 값이다. 1보다 크면 고위험 농축이다.
- **95% CI:** 같은 날짜 경주들의 의존성을 보존한 `rcDate` 블록 bootstrap 2,000회 분위수다.
- **관망(abstain):** 고위험 경주에서 시장 선호마 의존 전략의 거래를 보류하는 후보 규칙이다. 이 보고서는 ROI를
  계산하지 않았으므로 관망 규칙의 수익 개선을 주장하지 않는다.

## 1. 한눈에 보는 결론

- 학습 입력은 v11의 **141개 clean fundamental feature**를 경주별 mean/std/min/max로 집계한 뒤,
  train에서 상수인 열만 제거한 **486개**다.
- LightGBM 설정은 train 내부 시간순 holdout에서만 골랐다. 선택안은 `regularized_d4`이고,
  validation·test는 설정 선택에 쓰지 않았다.
- 확률 보정기는 train expanding-window OOF **1,896경주**만으로 적합했다.
- 10/20/30/40% 고위험·저위험 임계값은 validation 예측분포에서 숫자로 고정하고 test에 그대로 적용했다.
- 시장 기준은 경주별 하위절반 말의 정규화 역배당 확률 `sum(q)`이며 모델 피처가 아니다.
- 이 결과는 이변 위험 분류층의 탐색적 성능 근거일 뿐 ROI·수익성·배치 가능성을 입증하지 않는다.

## 2. 데이터·레이블·누수 통제

- 데이터: `C:\Users\user\source\repos\PythonApplication2\PythonApplication2\data\revised_v11_seoul_bugyeong_rank_clean_preprocessed`
- final reference: `C:\Users\user\Downloads\final (2).csv.gz`
- reference SHA-256: `964BD9A7AB7E36247FC5A7E5FFD04F9EC02439491B1DACAEF7918EB0AAE80195`
- reference 용도: **레이블, 최종 시장순위, 시장 baseline만**. 모델 행렬에는 결합하지 않았다.
- 최종배당 유효 기준: 경주의 모든 `winOdds`가 `1.0 < odds < 9999.0`, 모든 `q > 0`, 우승마 1두.
- 제외 경주: 1개. 상세는 `excluded_races.csv`.
- 시간 경계: train 20230805~20250511,
  valid 20250517~20251227,
  test 20251228~20260809.

## 3. train 내부 모델 선택

|candidate|fit_last_date|holdout_first_date|holdout_races|roc_auc|pr_auc|brier|selected|
|---|---|---|---|---|---|---|---|
|regularized_d4|20241229|20250103|639|0.547023|0.179058|0.113781|True|
|wide_d5|20241229|20250103|639|0.568259|0.169130|0.113831|False|
|balanced_d4|20241229|20250103|639|0.547499|0.182095|0.114027|False|
|compact_d3|20241229|20250103|639|0.540370|0.172677|0.114030|False|

## 4. test 전체·서울·부경 지표

모델은 보정 확률, 시장은 하위절반 말의 `q` 합이다. 각 CI는 날짜 블록 bootstrap이다.

|venue|source|races|roc_auc|roc_auc_ci_low|roc_auc_ci_high|pr_auc|pr_auc_ci_low|pr_auc_ci_high|brier|brier_ci_low|brier_ci_high|ece_10bin|ece_10bin_ci_low|ece_10bin_ci_high|observed_upset_rate|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|overall|calibrated_model|1100|0.540140|0.483690|0.596086|0.111693|0.088959|0.141495|0.095302|0.081349|0.110573|0.015853|0.003100|0.037212|0.106364|
|overall|market_baseline|1100|0.647573|0.590999|0.699428|0.209878|0.148902|0.281325|0.091882|0.078387|0.105661|0.027086|0.012821|0.046189|0.106364|
|seoul|calibrated_model|634|0.522628|0.451829|0.589390|0.115300|0.086710|0.161860|0.099774|0.080245|0.121239|0.012631|0.002509|0.040675|0.111987|
|seoul|market_baseline|634|0.669227|0.599101|0.736697|0.226062|0.155142|0.322911|0.095573|0.077174|0.116887|0.028399|0.013392|0.053620|0.111987|
|bugyeong|calibrated_model|466|0.568064|0.489667|0.646599|0.112351|0.080817|0.163198|0.089218|0.070735|0.110610|0.023375|0.005139|0.050889|0.098712|
|bugyeong|market_baseline|466|0.613820|0.511035|0.706853|0.186865|0.114290|0.288526|0.086862|0.068819|0.105537|0.025368|0.010168|0.053677|0.098712|

### 모델-시장 paired 차이

ROC/PR은 `모델-시장`, Brier는 `시장-모델`이라 양수면 모델 쪽이 좋다. CI가 0을 포함하면 차이를 확정하지 않는다.

|venue|races|roc_auc_difference_model_minus_market|roc_auc_difference_model_minus_market_ci_low|roc_auc_difference_model_minus_market_ci_high|pr_auc_difference_model_minus_market|pr_auc_difference_model_minus_market_ci_low|pr_auc_difference_model_minus_market_ci_high|brier_improvement_market_minus_model|brier_improvement_market_minus_model_ci_low|brier_improvement_market_minus_model_ci_high|
|---|---|---|---|---|---|---|---|---|---|---|
|overall|1100|-0.107433|-0.180872|-0.035687|-0.098185|-0.159501|-0.047708|-0.003420|-0.005820|-0.001271|
|seoul|634|-0.146599|-0.235728|-0.051770|-0.110762|-0.192267|-0.037595|-0.004201|-0.007307|-0.001341|
|bugyeong|466|-0.045756|-0.165187|0.077365|-0.074514|-0.163837|-0.010534|-0.002356|-0.005932|0.000982|

## 5. validation 고정 위험 임계값

분위수 산출에는 validation의 **확률만** 사용했다. test 레이블·확률분포로 임계값을 다시 맞추지 않았다.

|policy|direction|target_fraction|threshold|validation_realized_fraction|source_split|
|---|---|---|---|---|---|
|high_10pct|high|0.100000|0.149109|0.100554|valid|
|low_10pct|low|0.100000|0.093729|0.100554|valid|
|high_20pct|high|0.200000|0.138741|0.200185|valid|
|low_20pct|low|0.200000|0.101441|0.200185|valid|
|high_30pct|high|0.300000|0.130918|0.299815|valid|
|low_30pct|low|0.300000|0.107824|0.299815|valid|
|high_40pct|high|0.400000|0.124544|0.400369|valid|
|low_40pct|low|0.400000|0.113611|0.400369|valid|

## 6. 고·저위험 구간의 test 이변률과 lift

고위험 구간은 시장 선호마 전략의 관망 후보, 저위험 구간은 통과 후보로 해석할 수 있다. 이는 분류 진단이며
승식별 실제 배당·베팅 로그를 결합한 수익 검증이 아니다.

|venue|policy|direction|selected_races|realized_fraction|observed_upset_rate_all|observed_upset_rate_selected|lift|lift_ci_low|lift_ci_high|
|---|---|---|---|---|---|---|---|---|---|
|overall|high_10pct|high|116|0.105455|0.106364|0.086207|0.810492|0.416779|1.231837|
|overall|low_10pct|low|90|0.081818|0.106364|0.044444|0.417854|0.093415|0.854104|
|overall|high_20pct|high|228|0.207273|0.106364|0.109649|1.030889|0.742542|1.334257|
|overall|low_20pct|low|192|0.174545|0.106364|0.098958|0.930377|0.584164|1.306864|
|overall|high_30pct|high|356|0.323636|0.106364|0.115169|1.082781|0.817997|1.339683|
|overall|low_30pct|low|295|0.268182|0.106364|0.088136|0.828625|0.563373|1.097574|
|overall|high_40pct|high|464|0.421818|0.106364|0.120690|1.134689|0.925241|1.347165|
|overall|low_40pct|low|423|0.384545|0.106364|0.082742|0.777919|0.596355|0.976199|
|seoul|high_10pct|high|60|0.094637|0.111987|0.066667|0.595305|0.145422|1.144580|
|seoul|low_10pct|low|47|0.074132|0.111987|0.042553|0.379982|0.000000|0.973876|
|seoul|high_20pct|high|125|0.197161|0.111987|0.112000|1.000113|0.619975|1.419726|
|seoul|low_20pct|low|111|0.175079|0.111987|0.108108|0.965360|0.529276|1.463720|
|seoul|high_30pct|high|193|0.304416|0.111987|0.113990|1.017879|0.669813|1.399278|
|seoul|low_30pct|low|175|0.276025|0.111987|0.102857|0.918471|0.596924|1.280435|
|seoul|high_40pct|high|261|0.411672|0.111987|0.118774|1.060601|0.775525|1.341398|
|seoul|low_40pct|low|245|0.386435|0.111987|0.097959|0.874734|0.628084|1.138037|
|bugyeong|high_10pct|high|56|0.120172|0.098712|0.107143|1.085404|0.408736|1.815204|
|bugyeong|low_10pct|low|43|0.092275|0.098712|0.046512|0.471183|0.000000|1.209359|
|bugyeong|high_20pct|high|103|0.221030|0.098712|0.106796|1.081891|0.617102|1.650937|
|bugyeong|low_20pct|low|81|0.173820|0.098712|0.086420|0.875470|0.340295|1.497094|
|bugyeong|high_30pct|high|163|0.349785|0.098712|0.116564|1.180848|0.825125|1.560230|
|bugyeong|low_30pct|low|120|0.257511|0.098712|0.066667|0.675362|0.321666|1.075482|
|bugyeong|high_40pct|high|203|0.435622|0.098712|0.123153|1.247590|0.919602|1.557052|
|bugyeong|low_40pct|low|178|0.381974|0.098712|0.061798|0.626038|0.323750|0.958457|

## 7. test 전체 calibration 표

|source|bin|races|mean_predicted_probability|observed_upset_rate|
|---|---|---|---|---|
|calibrated_model|1|171|0.091870|0.093567|
|calibrated_model|2|929|0.127178|0.108719|
|market_baseline|1|386|0.077824|0.056995|
|market_baseline|2|653|0.135977|0.113323|
|market_baseline|3|52|0.230058|0.326923|
|market_baseline|4|5|0.348316|0.200000|
|market_baseline|5|4|0.454164|0.750000|

전체 split·경마장·source 10-bin 표는 `calibration_bins.csv`에 있다.

## 8. 이변 위험 피처 중요도

아래는 최종 train-only LightGBM gain을 141개 원 피처 수준으로 합산한 값이다. 시장배당 피처 중요도가 아니라
경주 단위 이변 레이블을 설명하는 중요도이며 인과효과가 아니다.

|rank|base_feature|normalized_gain|split_count|aggregate_columns|
|---|---|---|---|---|
|1|hr_last_finpct|0.042595|88|4|
|2|te_owName|0.042568|105|4|
|3|tr_winrate__z|0.038302|88|4|
|4|rating__z|0.031299|49|4|
|5|jk_winrate__z|0.029608|77|4|
|6|train_runs_14__z|0.027521|66|3|
|7|wg|0.026494|65|4|
|8|age|0.026433|61|4|
|9|train_sec_14|0.025661|62|4|
|10|wg_diff|0.023797|72|4|
|11|ow_plcrate|0.021702|64|4|
|12|tr_plcrate|0.021664|59|4|
|13|clinic_30d|0.021087|51|4|
|14|jk_plcrate|0.020178|56|4|
|15|train_days_14|0.020062|47|4|
|16|te_jkName|0.019721|54|4|
|17|ow_winrate|0.019343|51|4|
|18|te_trName|0.017901|48|4|
|19|oh_sex_암|0.017791|42|4|
|20|hr_rest_days__pr|0.017636|35|4|
|21|hr_last_dist|0.016971|45|4|
|22|hr_plcrate|0.016653|43|4|
|23|hr_winrate__z|0.016188|40|4|
|24|hr_rest_days|0.015155|42|4|
|25|tool_n|0.014828|39|4|

집계열별 전체 순위는 `aggregate_feature_importance.csv`, 원 피처 합산 순위는 `base_feature_importance.csv`다.

## 9. 재현·자체검증 근거

- 모델: `models/race_upset_lightgbm.txt`, `models/model_bundle.joblib`
- 레이블: `race_upset_labels.csv`
- 예측: `race_upset_probabilities.csv`
- 지표/CI: `metrics_by_split_venue_source.csv`, `model_vs_market_differences.csv`
- 임계값/lift: `validation_locked_risk_thresholds.csv`, `risk_band_lift.csv`
- 검증: `self_validation.json` — **PASS**
- 모든 산출물 해시: `artifact_manifest.json`

## 10. 제한과 다음 확증 단계

1. 현재 test가 이미 관찰된 기간이므로, 임계값과 모델을 고정한 뒤 그 이후 신규 경주를 한 번만 평가해야 한다.
2. 이변 위험층이 실제 수익을 개선하는지는 기존 승식별 OOS 베팅 후보에 validation-고정 관망 mask를 결합해
   동일 베팅 집합의 paired ROI/MDD/Sharpe 차이를 검증해야 한다. 여기서는 ROI를 만들거나 추정하지 않았다.
3. 최종배당은 결과 정의·시장 비교에만 사용했다. 실제 배치 시점에는 베팅 마감 직전 스냅샷 가용성과 지연을
   별도로 검증해야 한다.
4. feature importance는 연관성이다. 다음 신규 기간에서 경마장별 lift와 calibration이 같은 방향인지 우선 확인한다.
