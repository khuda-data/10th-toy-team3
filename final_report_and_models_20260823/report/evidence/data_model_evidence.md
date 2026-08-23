# 데이터·모델 근거 최종화 다이제스트

생성 시각: 2026-08-23 10:30 KST  
근거 루트: `C:\Users\user\source\repos\PythonApplication2\PythonApplication2`  
기계 판독 원본: `outputs/reports/final_comprehensive_research_report_20260823/evidence/data_model_evidence.json`

## 1. 범위와 판정 원칙

이 문서는 **기존 산출물만 읽어** 데이터셋, 전처리, 시계열 분할, 누수 감사, 실제 학습 이력, 순위·확률 지표, 시장 기준 비교를 최종화한 증거 다이제스트다. 이번 작업에서는 새 학습, 새 백테스트, 웹 검색, 신규 가설 검정을 하지 않았다. 수치는 인용한 CSV·JSON·로그에서만 옮겼고, 주요 파일의 현재 SHA-256을 읽기 전용으로 다시 계산했다.

핵심 판정은 다음과 같다.

- 현재 정규 데이터셋은 `revised_v11_seoul_bugyeong_rank_clean_preprocessed`다. 서울·부경 합계 56,456행, 5,343경주, 모델 피처 141개다.
- 숫자형·유한값·시간순 분할·훈련 구간 적합 전처리·상관성 제거·엄격한 시점 안전 타깃 인코딩 검사는 모두 기존 감사에서 통과했다.
- 전체 착순 학습 모델 6개가 실제 저장됐고 재로드 검증까지 통과했다. 최종 검증은 203/203 PASS다.
- 순위 성능의 단일 보편 우승 모델은 없다. CatBoost는 NDCG@5·Spearman·순위 MAE가 가장 좋고, LightGBM은 순서까지 맞춘 top-3 exact, Deep RankNet은 순서 무관 top-3 exact, XGBoost는 top-1 적중률이 가장 좋다.
- 시장 앵커 모델은 재사용 TEST에서 양의 ΔLL을 보인 경우가 있지만, 일반 전체착순 모델이 고른 단승·연승 후보는 TEST에서 시장 선택보다 적중률과 selected-event Brier가 모두 뒤졌다.
- 어떤 인용 결과도 **완전히 새로운 forward OOS에서 95% 신뢰구간 하한이 0을 넘는 절대 ROI**를 확립하지 못했다. 인용된 배포 판단은 모두 `stake=0` 또는 `NO_GO`다.

## 2. 현재 정규 데이터셋과 무결성

주요 근거는 전처리 매니페스트 `[E01]`, 전처리 검증 `[E02]`, 실행 데이터 감사 `[E04]`, 엄격한 타깃 인코딩 독립 검증 `[E05]`다.

| 분할 | 기간 | 행 | 경주 | 서울 행/경주 | 부경 행/경주 | 착순 상태 |
|---|---:|---:|---:|---:|---:|---|
| train | 2023-08-05~2025-05-11 | 33,416 | 3,158 | 19,617 / 1,891 | 13,799 / 1,267 | 완전 3,125, 동착·간격 33 |
| valid | 2025-05-17~2025-12-27 | 11,345 | 1,084 | 6,582 / 641 | 4,763 / 443 | 완전 1,076, 동착·간격 8 |
| test | 2025-12-28~2026-08-09 | 11,695 | 1,101 | 6,639 / 635 | 5,056 / 466 | 완전 1,078, 동착·간격 16, 0/NaN 착순 7 |

현재 데이터 파일 해시는 매니페스트 기록과 일치했다.

| 파일 | SHA-256 |
|---|---|
| `data/revised_v11_seoul_bugyeong_rank_clean_preprocessed/train_revised_v11_seoul_bugyeong_rank_clean_numeric_scaled.csv` | `B3228DDB785ED9B7F844BFBF76C8184DFAEEE6B35D412137E441CB205250161A` |
| `.../valid_revised_v11_seoul_bugyeong_rank_clean_numeric_scaled.csv` | `91791C80E391DC3125209E0CE73A8B8759D609CC7B2F58A64A45F91DAC2E67CA` |
| `.../test_revised_v11_seoul_bugyeong_rank_clean_numeric_scaled.csv` | `807589DB62BCD4F4D905435FD93F55DDAFFF2FF85788C435BB37358DA87146FE` |

전처리 검증 결과:

- 세 분할의 모델 피처는 모두 숫자형이며 NaN과 무한대가 각각 0개다.
- 피처 선택에 validation/test 통계를 사용하지 않았다.
- 최종 배당 풀 변수 `winAmt`, `plcAmt`, `totalAmt`, `log_winAmt`, `liq_per_horse`는 피처 선택 전에 제거했다.
- 훈련 구간 Pearson 절댓값 임계치는 0.95다. 제거 전 16쌍, 제거 후 0쌍이며, 최종 행렬의 직접 재검사 최대치는 0.9378611763704204 (`jk_winrate__z`, `jk_winrate__pr`)다.
- 고상관 피처 15개와 one-hot 기준 범주 `ageCond_연령오픈`을 제거했다. 제거 목록 전체는 기계 판독 JSON과 `[E01]`에 있다.
- 141개 비상수 숫자 피처 중 127개는 표준화 열이고 14개는 이진/단위구간 열이다. 모든 열을 억지로 평균 0·표준편차 1로 만들지 않은 것은 실패가 아니다. 특히 트리 모델은 표준 스케일링을 필수로 요구하지 않는다.

### 엄격한 타깃 인코딩

`te_jkName`, `te_trName`, `te_owName`, `te_rank`는 smoothing 20으로 계산됐다.

- train: 같은 날짜의 모든 말을 먼저 인코딩한 뒤 그 날짜 결과로 이력을 갱신한다. 즉, 오직 **엄격히 이전 날짜 결과**만 사용한다.
- valid/test: train 종료 시점 매핑만 사용하며 valid/test 결과로 갱신하지 않는다.
- 독립 재현 최대 오차: train `8.881784197001252e-16`, valid/test 각각 `4.440892098500626e-16`.
- 같은 날짜·미래 결과 반례 실험 최대 차이 0, valid/test 라벨 교란 반례 최대 차이 0.
- 독립 검증 60/60 PASS.

미관측 범주 fallback은 train prior로 정확히 재현됐다. TEST 미관측 비율은 기수 12.0735%, 조교사 7.9436%, 마주 3.4630%, rank 0%다. 이는 누수가 아니라 운영 시 신규 범주가 적지 않다는 **일반화 위험**이다.

## 3. Git 데이터와의 전처리 비교

`outputs/reports/dataset_preprocessing_comparison_20260823/evidence.json` `[E14]`의 기록상 승자는 현재 v11이다.

- 현재 v11: 56,456행, 141 모델 피처, 비숫자·결측·무한대 0, 훈련 Pearson ≥0.95 쌍 0.
- Git 정규 파이프라인: 원천 32,888행, 등록 pre-race 피처 112, 런타임 변환 후 554열이며 변환 후 비숫자·NaN·무한대는 0이다. 다만 원천 train 숫자 피처에는 |r|≥0.95가 25쌍, 사실상 1인 쌍이 4개이고 상관 pruning은 구현되지 않았다.

단, 비교는 완전히 동형이 아니다. Git 쪽은 interim source와 런타임 transformer이고 v11은 물리화된 모델 행렬이다. Git의 트리 파이프라인이 스케일링을 생략한 것은 설계상 가능하다. Pearson pruning만으로 비선형 중복이나 다중공선성이 모두 사라졌다고 말할 수도 없다.

## 4. 시계열 분할과 누수 감사

시간순은 `train < valid < test`로 엄격하며, 분할 사이 entry/race overlap은 0으로 감사됐다. 전체착순 X에는 배당·결과·착순 라벨 열이 없었다. 시장 앵커에서는 배당이 일반 X가 아니라 경주 내 역단승배당을 정규화한 `q`의 고정 `log(q)` offset/base margin으로만 들어간다. q 항등식 최대 오차는 `2.7755575615628914e-16`이다.

다만 다음 경계는 반드시 보존해야 한다.

1. v11 TEST는 이미 여러 연구에서 확인·재사용됐다. 양의 TEST ΔLL은 독립 확증이 아니다.
2. H10A `fresh64`도 이미 관측된 optional continuation이며 사후 분석이다.
3. H11D 이후 H13/H14/H17은 같은 late-VALID 28일을 재사용했다.
4. H11D는 TEST 행을 0개 파싱했지만, 분석 전 `Get-FileHash`가 파일 전체 바이트를 스트리밍했다. TEST를 지표에 사용하지는 않았으나 엄격한 물리 I/O 격리는 0이 아니다. `[E45]`
5. 여러 분석의 q와 배당 구간은 마감 배당이다. 실제 의사결정 시점 가격, 슬리피지, 풀 충격은 해결되지 않았다.

## 5. 전체 착순 모델의 실제 학습 증거

학습 소스는 `src/training/train_full_rank_models.py` `[E23]`, SHA-256 `37E0C01927AC8AFFEE364813526BC4BEB819B737065949E2D6BB3A7E860011DF`다. 최종 검증 `[E06]`은 모델 6개, 모델별 TEST 예측 11,695행, 재로드 점수 최대 차이 `4.440892098500626e-16`, 203/203 PASS를 기록한다.

| 모델 | 실제 학습 방식 | 핵심 설정/학습량 | 저장·로그 증거 |
|---|---|---|---|
| Random Forest | 정상화 착순 백분위를 회귀, 낮을수록 상위 | 500 trees, depth 14, leaf 8, seed 42, 33,416행 | `[E24]`, `[E32]` |
| XGBoost | 경주 qid의 `rank:ndcg`, 전체 착순 relevance | 최대 2,000, early stop 120, 최종 448 iteration | `[E25]`, `[E33]` |
| LightGBM | 경주 그룹 LambdaRank | 최대 3,000, early stop 150, 최종 54 iteration | `[E26]`, `[E34]` |
| CatBoost | YetiRankPairwise, 경주 group_id | 최대 2,500, 최적 zero-based 767 | `[E27]`, `[E35]` |
| Deep RankNet | 경주 내 서로 다른 착순 모든 쌍의 pairwise softplus | 141→256→128→1, 3 seeds, CPU, AdamW | `[E28]`~`[E30]`, `[E36]` |
| Plackett-Luce Neural | 모든 착순 위치의 PL likelihood | 완전 착순 3,125경주, 비엄격 33경주 제외, CPU | `[E31]`, `[E37]` |

각 로그에는 `FULL RANK MODEL TRAINING COMPLETE`가 남아 있고, 모델 파일 자체와 reload 재현 검사가 모두 존재한다. 따라서 “코드만 만들고 학습을 생략했다”는 상태는 아니다.

## 6. TEST 순위 지표

일반 순위 지표는 유효 착순 1,094경주·11,628행에서, top-3 exact는 완전 1~N 착순 1,078경주에서 계산됐다. 7개 무효 착순 경주와 16개 동착·간격 경주 때문에 분모가 다르다.

| 모델 | NDCG@5 | Spearman | 순위 MAE | Pairwise | Top-1 | Top-3 recall | Top-3 순서 일치 | Top-3 순서 무관 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | **0.778036** | **0.455303** | **2.439987** | **0.671415** | 0.300731 | 0.510969 | 0.013915 | 0.072356 |
| Deep RankNet | 0.775170 | 0.444988 | 2.467096 | 0.666214 | 0.311700 | 0.508836 | 0.015770 | **0.077922** |
| XGBoost | 0.773241 | 0.436575 | 2.491832 | 0.663658 | **0.315356** | 0.509522 | 0.014842 | 0.073284 |
| Plackett-Luce | 0.771865 | 0.436825 | 2.481820 | 0.664451 | 0.290676 | 0.508836 | 0.015770 | 0.074212 |
| LightGBM | 0.768468 | 0.420647 | 2.538521 | 0.657328 | 0.303473 | 0.497182 | **0.017625** | 0.068646 |
| Random Forest | 0.768381 | 0.428041 | 2.511839 | 0.660766 | 0.304388 | 0.502133 | 0.012059 | 0.073284 |

경주별 softmax 확률 합 최대 오차는 `4.440892098500626e-16`이고 모든 확률은 유한하다. validation만으로 맞춘 온도는 RF 0.117370, XGB 0.672135, LGB 0.472781, Cat 0.627690, Deep 0.631919, PL 0.556552다. `[E08]`

## 7. 일반 전체착순 모델과 시장 선택 비교

`[E09]`의 TEST 단승·연승 직접 비교에서 시장 선택 적중률은 단승 0.376623, 연승 0.679775다.

| 모델 | 단승 적중 | 시장 대비 selected-event Brier 개선 | 연승 적중 | 시장 대비 selected-event Brier 개선 |
|---|---:|---:|---:|---:|
| RF | 0.303340 | -0.012597 | 0.596442 | -0.022197 |
| XGB | 0.314471 | -0.012492 | 0.618914 | -0.013989 |
| LGB | 0.303340 | -0.014875 | 0.603933 | -0.019462 |
| Cat | 0.299629 | -0.017148 | 0.605805 | -0.024700 |
| Deep | 0.311688 | -0.015409 | 0.606742 | -0.029049 |
| PL | 0.290353 | -0.016950 | 0.600187 | -0.025324 |
| Ensemble | 0.318182 | -0.011692 | 0.618914 | -0.013843 |

모든 모델과 앙상블이 단승·연승 적중률에서 시장보다 낮고, 같은 모델 선택 사건에 대한 Brier 개선도 모두 음수다. 이 결과는 “착순 모델이 곧 시장을 이기는 확률 모델”이 아님을 직접 보여준다.

## 8. v11 시장 앵커와 ΔLL

시장 앵커는 141개 X에 배당 열을 섞은 모델이 아니다. `q = (1/winOdds) / Σ_race(1/winOdds)`를 계산한 뒤 `log(q)` 계수를 1로 고정하고 일반 피처가 잔차만 보정한다. 조건부 로짓의 L2는 100, Base Margin 트리 수는 188로 validation에서 고정한 뒤 train+valid로 TEST 모델을 재적합했다. 저장 모델 2개도 현재 해시와 validation 기록이 일치한다. `[E10]`~`[E12]`, `[E38]`, `[E39]`, `[E49]`, `[E50]`

ΔLL은 `시장 경주당 winner NLL - 모델 경주당 winner NLL`이라 양수가 모델 우위다.

| 분할 | 모델 | ΔLL/경주 | 일별 블록 95% CI | 모델 Brier | 시장 Brier | 모델 Top-1 | 시장 Top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| valid | 조건부 로짓 | 0.005312 | [-0.007519, 0.017481] | 0.072713 | 0.072924 | 0.378229 | 0.379151 |
| valid | Base Margin | 0.021833 | [0.013618, 0.030266] | 0.072195 | 0.072924 | 0.383764 | 0.379151 |
| valid | 앙상블 | 0.017756 | [0.008532, 0.027182] | 0.072329 | 0.072924 | 0.380996 | 0.379151 |
| reused test | 조건부 로짓 | 0.013974 | [0.002318, 0.025063] | 0.072126 | 0.072449 | 0.383288 | 0.378747 |
| reused test | Base Margin | 0.013083 | [0.006614, 0.019647] | 0.072203 | 0.072449 | 0.375114 | 0.378747 |
| reused test | 앙상블 | 0.017194 | [0.009507, 0.025014] | 0.072047 | 0.072449 | 0.376022 | 0.378747 |

확률 분포 측면의 시장 대비 개선은 보이지만, TEST는 이미 재사용됐고 Top-1은 모델마다 시장보다 높기도 낮기도 하다. ΔLL 양수는 수익 보장이 아니다.

## 9. 시장 기준 7개 승식 ROI

`[E13]`은 TEST에서 시장 선호 선택 한 건씩의 기준 ROI를 기록한다. 최종 배당/환급 자료를 이용한 시장 기준이며 별도 배포 전략이 아니다.

| 승식 | 경주 | 적중률 | ROI | 일별 블록 95% CI |
|---|---:|---:|---:|---:|
| 단승 | 1,077 | 0.376973 | -17.604% | [-23.315%, -11.516%] |
| 연승 | 1,004 | 0.668327 | -15.339% | [-18.659%, -12.100%] |
| 복승 | 1,077 | 0.198700 | -8.459% | [-20.841%, 5.214%] |
| 쌍승 | 1,077 | 0.116063 | -13.900% | [-28.787%, 1.447%] |
| 삼복승 | 1,077 | 0.111421 | -24.596% | [-37.834%, -11.213%] |
| 삼쌍승 | 1,077 | 0.030641 | -33.491% | [-55.211%, -9.827%] |
| 복연승 | 1,077 | 0.365831 | -20.743% | [-27.371%, -14.222%] |

복합 승식에서 관련 model-vs-market event probability는 실제 조합 풀 확률이 아니라 q에서 유도한 proxy다. 공식 배당은 정산을 식별할 뿐, proxy를 실제 시장 EDGE로 바꾸지는 않는다.

## 10. 후속 시장 상대 연구의 실제 학습과 결과

### H5~H7

- H5 adaptive: 조건부 로짓 실제 fit 32회, 모델 파일 4개, rolling OOS 2,185경주. 시장 대비 ΔLL 0.009587, CI [0.002457, 0.016667]. 8~15배 연승 366건 ROI 2.432%지만 CI [-13.405%, 18.798%]로 절대 이익 미확정. 같은 경주의 시장 선택 대비 ROI 차이는 23.443%p, CI [6.197%, 41.767%]. `[E18]`
- H5 fixed-lock: 실제 fit 32회, 모델 파일 8개, VALID에서 edge threshold 0.0709603844를 고정. TEST rolling ΔLL 0.012046, CI [0.003180, 0.021159]. 연승 200건 ROI 1.300%, CI [-18.647%, 21.961%]; 시장 대비 31.800%p, CI [10.510%, 53.762%]. Kelly 총수익 -0.806%. `[E19]`
- H6: calibrator 4개 실제 학습, 후보 관측 961개. 보정 후 uncalibrated 대비 LL 개선 0.005231, 원 p=0.0223이나 Holm p=0.0684로 다중검정 후 유의하지 않다. Kelly 수익은 미보정 -0.806%, 보정 -18.827%. `[E20]`
- H7: 조건부 로짓 40회, ablation 32회, 모델 파일 8개. `clinic_30d` 증분 ΔLL은 global 0.001300, partial8 0.000943이며 두 CI 모두 0을 포함한다. pre-race 생성 시점 정의가 독립적으로 확인되기 전 배포 차단이다. `[E21]`

따라서 H5는 “시장보다 덜 나쁜 방어적 선택”의 신호는 있지만 절대 수익은 확정하지 못했다. H6/H7은 추가 복잡성이 강건한 개선으로 이어지지 않았다.

### H10A·H11B

H10A는 `age`, `age__pr`, `oh_sex_암`, `wgBudam_chg`, `wgBudam__pr`, `te_owName` 여섯 피처의 조건부 로짓을 6회 실제 fit하고 모델 파일 3개를 남겼다. VALID ΔLL 0.001585의 CI는 0을 포함하고, 재사용 TEST ΔLL 0.005422는 양수지만 독립 확증이 아니다. fresh64 ΔLL 0.010457도 일별 CI [-0.001784, 0.024654], 6일 exact sign-flip p=0.078125다. 배포는 `NO_GO`, stake=0이다. `[E22]`

H11B는 같은 6개 피처의 시장 잔차를 XGBoost, LightGBM, 3-seed Deep MLP로 학습했다.

- 최종 성공 실행: XGB 3회, LGB 3회, Deep 9회 = 15 fit. 기존 M6 3개는 로드했으며 재학습하지 않았다.
- 중단 실행 fit 18회를 합치면 누적 fit operation은 33회다. 33개의 최종 독립 모델을 뜻하지 않는다.
- custom NLL gradient `p-y` 최대 유한차분 오차 `2.4037e-10`, diagonal Hessian `p(1-p)` 오차 `2.6313e-08`로 각각 1e-6, 1e-5 허용치 이내다.
- 첫 독립 validator는 Deep float32 실행 경로 차이 최대 `6.80599e-09` 때문에 27/28 실패했다. validator만 단일 CPU thread·batch 4096으로 맞췄고 허용치를 완화하거나 모델·예측을 바꾸지 않은 뒤 28/28 PASS, 최대 차이 `3.33067e-16`이 됐다. `[E43]`
- VALID에서 모든 모델의 ΔLL CI가 0을 포함했다. 재사용 TEST에서는 M6와 Deep만 CI 하한이 양수였고, XGB·LGB는 ΔLL이 음수, equal ensemble CI는 0을 포함했다.
- stake=0이다.

실행 로그 `[E41]`에는 두 실패도 보존돼 있다. 첫 실행은 15 fit 후 phase logging TypeError, 두 번째는 XGB 3 fit 후 LightGBM Dataset construct 전 init-score 검사 오류, 마지막 실행은 15 fit 후 정상 동결됐다.

### H11D·H13·H14·H17

이 계열은 late-VALID 28일·340경주를 반복 사용한 적응적 연구이므로 독립 확증이 아니다.

| 연구 | 실제 fit/선택 | late28 시장 대비 ΔLL | 95% CI | 핵심 해석 |
|---|---:|---:|---:|---|
| H11D alpha shrinkage | 개발 64일에서 α=1.05 | 0.002289 | [-0.003839, 0.008967] | 유의하지 않음, all-valid 배포 α=1.06은 독립 평가 없음 |
| H13 비음수 log pool | optimizer 8회, λ=0 | 0.002006 | [-0.004539, 0.009043] | first64 가중치 합 1.145로 음의 시장 지수 위험 |
| H14 convex pool | optimizer 3회, Σw≤1 | 0.002263 | [-0.003475, 0.008439] | first64→all92 L1 거리 0.9128, 가중치 불안정 |
| H17 ridge convex | optimizer 14회, 1-SE λ=1 | 0.0000837 | [-0.0000295, 0.0002075] | first64 잔차 가중치 합 0.01923, 거의 시장으로 수축 |

H17의 결과는 흥미롭지만 낙관적 결론이 아니다. 강한 정규화를 걸면 잔차 모델이 거의 시장 q로 되돌아간다는 뜻이며, 자유로운 pooling의 apparent gain 일부가 불안정한 가중치에서 나왔을 가능성을 지지한다.

## 11. 충돌·오래된 결과·실패를 어떻게 처리해야 하는가

1. `Project_Reserach.txt`와 `Report_summary.txt`의 일부 다음 작업 문구는 H10A~H17 이전에 생성돼 오래됐다. 최종 수치는 날짜가 박힌 primary JSON/CSV/log를 우선한다.
2. revised_v7/v10의 137-feature, 서울-only 또는 final-pool 누수 가능 결과는 역사적 산출물이다. 현재 데이터 무결성 주장은 v11 141-feature 서울+부경 clean 자료만 사용한다.
3. TEST 양의 ΔLL을 독립 확증으로 승격하지 않는다.
4. H5 시장 상대 우위와 절대 ROI를 혼동하지 않는다. 전자는 CI 하한이 양수여도 후자는 0을 포함한다.
5. 복합 승식 q-derived proxy를 실제 조합 풀 확률로 표현하지 않는다.
6. H11B 누적 fit 33회와 최종 성공 fit 15회를 분리한다.
7. H11B validator 최초 실패와 H11D 물리 바이트 I/O 사건을 삭제하지 않는다.
8. 순위 지표 분모 1,094경주와 top-3 exact 분모 1,078경주를 혼용하지 않는다.
9. ΔLL, 적중률, ROI는 서로 다른 목적함수다. 한 지표의 개선이 다른 지표의 개선을 보장하지 않는다.

## 12. 최종 결론

사실로 확인되는 성과는 세 가지다.

1. 현재 v11 데이터는 이 프로젝트에서 가장 잘 감사된 데이터다. 서울·부경을 포함하며 숫자형·결측·무한대·상관 pruning·시계열·엄격 타깃 인코딩 검증 근거가 있다.
2. Random Forest, XGBoost, LightGBM, CatBoost, Deep RankNet, Plackett-Luce와 후속 시장 잔차 모델은 실제로 학습·저장·재로드됐다.
3. 시장 앵커와 일부 rolling specification은 과거 구간에서 시장보다 나은 winner 확률 손실 또는 시장 선택 대비 방어적 연승 결과를 보였다.

그러나 아직 사실로 확정되지 않은 것은 더 중요하다.

- 전체착순 모델은 TEST 단승·연승 후보 선택에서 시장보다 강하지 않았다.
- 시장 앵커의 양의 ΔLL은 재사용 TEST에 의존한다.
- H5의 절대 ROI는 95% 구간이 0을 포함한다.
- 복합 승식의 실제 의사결정 시점 시장 확률과 가격은 확보되지 않았다.
- 완전히 새로운 forward OOS에서 독립 재현된 수익 우위가 없다.

따라서 현재 증거에 맞는 문장은 **“일부 사양에서 시장이 놓친 작은 확률 신호와 손실 방어 가능성은 관측됐지만, 실전 수익 시스템은 독립 확증되지 않았다”**이다. 그보다 강한 표현은 기존 파일이 지지하지 않는다.

## 13. 주요 근거 색인

전체 50개 근거의 상대·절대 경로, 바이트 수, SHA-256은 동명의 JSON `evidence_file_index`에 있다. 특히 다음 파일을 우선 확인하면 된다.

- `[E01]` `data/revised_v11_seoul_bugyeong_rank_clean_preprocessed/preprocessing_manifest.json`
- `[E02]` `data/revised_v11_seoul_bugyeong_rank_clean_preprocessed/preprocessing_validation.json`
- `[E05]` `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/strict_te_independent_validation.json`
- `[E06]` `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/final_validation.json`
- `[E07]` `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/model_metric_comparison.csv`
- `[E09]` `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/all_bet_types/bet_type_edge_analysis/bet_type_model_vs_market_summary.csv`
- `[E12]` `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/delta_ll_metrics.csv`
- `[E13]` `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/market_baseline.csv`
- `[E17]` `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/experiment_summary.json`
- `[E18]` `outputs/reports/rolling_origin_market_challenger_20260823/experiment_summary.json`
- `[E19]` `outputs/reports/fixed_lock_rolling_stability_20260823/experiment_summary.json`
- `[E43]` `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/independent_validation_incident_20260823.json`
- `[E45]` `outputs/reports/h11d_market_anchor_shrinkage_preregistered_20260823/physical_byte_io_disclosure.json`
- `[E48]` `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/experiment_summary.json`
