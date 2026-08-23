# 최종 종합 HTML 보고서 생성 명세

## 0. 명세의 목적과 절대 경계

이 문서는 `outputs/reports/final_comprehensive_research_report_20260823/index.html` 생성기가 이미 존재하는 연구 산출물만 사용해 최종 보고서를 만들도록 고정하는 구현 명세다. **새 학습, 새 예측, 재튜닝, 새 백테스트, 사후 target 정산, 웹 재검색을 허용하지 않는다.**

생성기의 최상위 판정은 다음 문장을 바꾸지 않는다.

> 현재 근거에서는 배치 가능한 수익 전략이 확인되지 않았고 실베팅 권고액은 0이다. 시장 앵커 계열에서 반복된 확률정보 개선과 재사용 OOS 8~15배 연승의 상대적 손실 방어 신호는 존재하지만, 완전 신규 forward OOS의 확정된 절대수익은 없다.

보고서가 구분해야 할 여섯 증거 상태:

1. `historical_or_reused_test`: 학습 시점 기준 OOS일 수 있으나 이미 여러 차례 관찰된 TEST.
2. `retrospective_rolling_oos`: 시간순 적합은 지켰지만 날짜가 기존 연구에 노출된 rolling OOS.
3. `later_date_optional_continuation_fresh64`: initial27 관찰 뒤 fresh37을 더한 later-date pooled64.
4. `valid_selection_only`: VALID에서 문턱·정책만 고정, target 성과는 아직 없음.
5. `prospective_preregistered_unsettled`: 사전고정은 됐으나 결과·환급 미정산.
6. `synthetic_contract_validation`: 수식·검증기·게이트만 시험한 합성 검증으로 성과 근거가 아님.

점추정치, 95% 신뢰구간, 다중검정, 실행가격 여부를 분리한다. 측정되지 않은 값은 `미측정`, 구조적으로 식별 불가능한 값은 `식별 불가`, 아직 결과가 없는 값은 `미정산`으로 표시한다. 이 세 표현을 0이나 실패값으로 대체하지 않는다.

## 1. 입력 계약

### 1.1 최종화 다이제스트 3종

생성기는 아래 JSON을 수치의 우선 원천으로 사용하고, 같은 이름의 Markdown은 설명·문맥 검토용으로만 사용한다.

| 역할 | JSON | 설명본 |
|---|---|---|
| 데이터·전처리·모델·학습증거 | `evidence/data_model_evidence.json` | `evidence/data_model_evidence.md` |
| 베팅·EDGE·ROI·위험 | `evidence/betting_edge_evidence.json` | `evidence/betting_edge_evidence.md` |
| 프로젝트 연대기·논문·현업자료 | `evidence/chronology_literature_evidence.json` | `evidence/chronology_literature_evidence.md` |

경로 기준은 최종 보고서 폴더 `outputs/reports/final_comprehensive_research_report_20260823/`다. 생성 전에 6개 파일 존재 여부를 검사한다. JSON parse가 실패하거나 필수 top-level key가 없으면 조용히 보간하지 말고 빌드를 실패시킨다.

현재 확인된 필수 JSON key:

- data/model: `executive_verdict`, `canonical_dataset`, `chronology_and_leakage`, `actual_training_inventory`, `rank_and_probability_metrics`, `market_baseline_comparison`, `conflicts_stale_and_failed_results`, `honest_limits`, `evidence_file_index`.
- betting: `overall_status`, `executive_findings`, `definitions`, `evidence_scope_taxonomy`, `v11_locked_full_rank_seven_bets`, `v11_market_anchor`, `v11_upset_feature_and_top_fraction`, `odds_8_15_evidence`, `h5_to_h9`, `h10_to_h18`, `prospective_20260823_snapshot_at_inventory`, `deep_and_stack_defensive_evidence`, `historical_superseded_or_nonconfirmatory`, `price_fee_kelly_and_slippage_limitations`, `failures_and_limits`, `quantitative_insights`, `chart_ready_existing_datasets`, `artifact_manifest`.
- chronology/literature: `status`, `scope`, `executive_evidence_boundary`, `chronology`, `hypothesis_outcomes`, `incidents_and_corrections`, `literature_and_methodology`, `terminology`, `key_generated_files`, `reusable_charts_and_assets`, `report_and_validation_history`, `metrics_not_measured`, `final_handoff`. 외부자료 URL·자료유형·프로젝트 연대기·주장 경계를 가진 실제 레코드만 사용한다. 이 digest는 이번 finalization에서 외부 원문을 재조회하지 않았다고 명시하므로, 원문 확인일이 없는 자료에 날짜를 만들어 붙이지 않고 `저장 카탈로그 재사용·이번 작업에서 미갱신`으로 표시한다.

### 1.2 원시 근거와 우선순위

동일 수치가 충돌하면 다음 순서를 적용한다.

1. 최종화 JSON 안에서 `scope`, `status`, `boundary`가 명시된 현재 판정.
2. JSON이 가리키는 독립 검증 CSV/JSON.
3. 현재 v11 complete report의 snapshot/manifest.
4. 과거 보고서는 오직 `historical/superseded/nonconfirmatory` 표에만 사용.

어떤 경우에도 과거 높은 ROI를 현재 확증 수치로 승격하지 않는다. `base_margin_profit_20260821`의 동일데이터 정책선택 결과, v7 137피처 누수 경로, optional fresh64, 재사용 late-VALID는 명시된 경계를 유지한다.

### 1.3 스타일 기준

스타일과 정보구조의 기준은 `../revised_v11_seoul_bugyeong_full_rerun_20260822/complete_report/index.html`이다. 재사용할 패턴:

- 좌측 sticky 목차와 본문 1열, 모바일에서는 단일열.
- hero, KPI strip, section, card/grid, callout, badge, table-wrap, details/summary.
- light/dark CSS variables와 `prefers-color-scheme`.
- 외부 라이브러리 없는 단일 HTML, chart는 접근 가능한 inline SVG.
- 긴 증거 표는 `<details>`로 접고, 핵심 판정은 접지 않는다.
- print CSS에서 sidebar를 숨기고 표·차트의 페이지 분할을 억제한다.
- 색상만으로 판정하지 않고 항상 `확인/미확인/미정산/식별 불가/NO-GO` 텍스트를 함께 둔다.

권장 기존 색상 토큰: background `#f4f6fb`, paper `#fff`, ink `#172033`, muted `#5f6b7a`, line `#d9dfeb`, blue `#315da8/#5b8def`, green `#14855f`, red `#b4523a`, amber `#9a6710`, purple `#6846a5`. 성공색은 오직 통계·증거 기준을 실제로 통과한 항목에만 사용한다.

## 2. 최종 HTML의 섹션 구조

### `#summary` — 0. 이번 분기 작업 요약 및 최종 판정

hero 제목은 `시장이 얼마나 틀리는가 — 최종 종합 연구 보고서`로 한다. 바로 아래에 사용자 지침에 맞춰 `[이번 분기 작업 요약 및 변경 사항]`을 5줄 이내로 둔다.

필수 KPI:

- 정규 데이터: 56,456행 / 5,343경주 / 서울+부경.
- 모델 입력: 141피처, ordinary X의 배당·결과 열 0개.
- 실제 전체착순 모델: 6개, reload·최종검증 203/203 PASS.
- 확정된 배치 수익 승식: 0개, stake 0.

그 아래 네 가지 문장을 판정 카드로 둔다.

1. 데이터·학습 무결성은 확인됨.
2. 확률정보 우위는 일부 재사용 OOS에서 관찰됨.
3. 절대수익·실행가능성은 미확인.
4. 2026-08-23 prospective는 다이제스트의 고정시각 현재 미정산.

source: `data_model_evidence.json/executive_verdict`, `betting_edge_evidence.json/overall_status|executive_findings|prospective_20260823_snapshot_at_inventory`.

### `#glossary` — 1. 지표와 공식

보고서 맨 앞에서 쉬운 뜻, 산식, 좋은 방향, 이 프로젝트의 용도를 한 표로 제공한다. 최소 항목:

- 시장확률 `q_ri=(1/O_ri)/Σ_j(1/O_rj)`.
- 확률 EDGE `p_ri-q_ri`.
- 실현 EDGE `I(event)-q`.
- EV `pO-1`, break-even `1/O`.
- DeltaLL/race `mean[log p(winner)-log q(winner)]`.
- Brier improvement `Brier_market-Brier_model`.
- NDCG@5, Spearman, 순위 MAE, top-3 순서일치, top-3 무순서일치.
- ROI, MDD, 비연율화 `sqrt(n)` Sharpe, hit rate, 평균 적중환급.
- Full/Fractional Kelly `f*=(pO-1)/(O-1)`.
- daily-block CI, Holm, BH-FDR, slippage/payout haircut.
- Base Margin과 market anchor: 일반 X 피처가 아니라 `log(q)` offset 위 잔차학습임을 명시.

source: `betting_edge_evidence.json/definitions`, `data_model_evidence.json/chronology_and_leakage.market_anchor_role`. 생소한 OSS, OOS, CAW, pari-mutuel, closing price는 가까운 각주와 glossary anchor를 모두 제공한다.

### `#question` — 2. 연구 질문과 증거 사다리

주 질문을 세 단계로 분리한다.

1. 모델이 시장보다 확률을 더 잘 추정하는가?
2. 그 차이가 실행 가능한 가격에서 양의 EV가 되는가?
3. 공제·슬리피지·불확실성·다중검정 후 장기 자금곡선이 견디는가?

증거 범주 6개를 위계 다이어그램으로 표시한다. `prospective settled forward OOS`가 최상단이지만 현재 측정값이 없다는 사실을 명시한다. synthetic PASS가 ROI PASS로 이동하는 화살표를 그리지 않는다.

### `#literature` — 3. Benter 이후 연구와 현업 구조

chronology/literature 다이제스트에서 확인된 문헌과 회사·기관 자료만 사용한다. 네 묶음으로 구성한다.

1. Benter/시장앵커/확률보정/Kelly의 고전적 구조.
2. 최근 LTR·누수통제·중간배당 궤적·exchange/attention 연구.
3. 현업 회사의 실제 수익원: 모델 초과수익뿐 아니라 리베이트, 수수료, B2B 데이터·토트·리스크관리, 풀 규모/유동성.
4. 문헌 방법을 현재 프로젝트가 적용/미적용/차단한 매핑.

논문 성능 숫자와 우리 성능을 직접 막대그래프로 비교하지 않는다. 데이터, 기간, 시장, 공제, 분모가 달라 잘못된 비교가 된다. 대신 `문헌 주장 → 필요한 조건 → 현재 적용 → 남은 차이` 표를 사용한다. 회사 자체 자료는 `회사 주장/공식 상품설명`, SEC·연차보고서는 `공시`, 프리프린트는 `동료심사 전` badge를 붙인다.

### `#chronology` — 4. 프로젝트 연대기와 결과의 승격·폐기

`chronology_literature_evidence.json/chronology`의 실제 15개 타임라인 레코드를 사용해 다음 흐름을 보인다.

`v5~v9 서울-only/구 피처 → v10 서울+부경 → v11 clean 141피처 → 시장앵커·7승식·이변 → H5~H10 retrospective/later-date → H11~H18 prospective 준비`

각 단계는 실제 필드 `order`, `period`, `phase`, `question`, `work`, `result`, `scientific_status`, `confidence`, `evidence`를 사용한다. `v7 137 final-pool leakage`, `과거 동일데이터 8~15 top5`, `fresh64 optional continuation`, `H11B first-attempt interruption`, `H11D/H13/H14/H17 late28 재사용`, `prospective unsettled`을 누락하지 않는다. 별도의 H1~H18 원장은 `hypothesis_outcomes`의 `id`, `hypothesis`, `outcome`, `fact`, `confidence`, `evidence`를 사용해 보강한다.

### `#data` — 5. 정규 데이터셋, Git 비교, 전처리 무결성

하위 섹션:

1. 원천 체인과 착순 label reference의 역할.
2. train/valid/test 기간·행·경주·서울/부경 분포.
3. 숫자형 변환, 결측·무한대, train-only imputation/scaling.
4. Pearson `|r|≥0.95` 제거 전 16쌍/후 0쌍, 최종 최대 0.937861.
5. 최종풀 5개 열 제거.
6. 엄격 날짜 target encoding 60/60 PASS와 unseen fallback 비율.
7. Git 데이터와의 비동형 비교: 현재 v11이 기록상 더 완결됐지만 raw/transform runtime 차이를 보존.
8. 착순 이상경주: 일반 rank metric 1,094경주, exact top-3 1,078경주라는 분모 차이.

source: `data_model_evidence.json/canonical_dataset`. dataset PASS를 ‘모든 종류의 누수·비선형 중복 부재’로 확대 해석하지 않는다.

### `#chronology-audit` — 6. 시계열 분할과 누수 감사

train `<` valid `<` test 타임라인, cross-split entry/race overlap 0, 전처리 train-only, ordinary X의 배당·결과 열 0을 시각적으로 보인다. 시장 q는 ordinary feature가 아니라 closing win odds의 race-normalized offset이라는 예외를 명시한다.

필수 경고:

- v11 TEST 재사용.
- fresh64 optional continuation.
- late28 VALID 재사용.
- H11D는 TEST row parse 0이나 whole-file hash로 물리적 TEST-region bytes를 읽음.
- closing odds는 T-5 executable quote가 아님.

source: `data_model_evidence.json/chronology_and_leakage|conflicts_stale_and_failed_results`.

### `#training` — 7. 실제 학습한 모델과 재현 증거

전체착순 6개 모델 각각에 모델명, 목적함수, 경주 grouping, 실제 fit rows, iteration/seed, 저장물, 완료로그, reload status를 표로 둔다.

- Random Forest rank regression.
- XGBoost `rank:ndcg`.
- LightGBM LambdaRank.
- CatBoost YetiRankPairwise.
- 3-seed Deep RankNet.
- Plackett-Luce neural.

H11B는 별도 하위표에 M6/XGB/LGB/deep MLP/equal ensemble, fit operation 33, 성공 final fit 15, interrupted operation 18, 확률합 오차, reload 오차, target outcome 미열람을 기록한다. 최초 실패를 지우지 말고 보존한다. source: `data_model_evidence.json/actual_training_inventory`와 각 `evidence_file_index` 항목.

### `#rank` — 8. 전체착순 지표

단일 우승 모델이라는 표현을 금지하고 지표별 우승을 제시한다.

- NDCG@5·Spearman·rank MAE: CatBoost.
- ordered top-3 exact: LightGBM.
- unordered top-3 exact: Deep RankNet.
- top-1 hit: XGBoost.

모든 6모델의 TEST 표에는 NDCG@5, Spearman, rank MAE, pairwise accuracy, top1, top3 recall, ordered/unordered exact를 둔다. 1,094/1,078 분모 차이를 표 바로 아래에 표시한다.

source CSV: `../revised_v11_seoul_bugyeong_full_rerun_20260822/model_metric_comparison.csv`.

### `#market` — 9. 시장 기준선과 앵커 확률품질

다음 두 결과를 같은 화면에 두되 동일한 판정으로 합치지 않는다.

1. ordinary full-rank 모델의 단승/연승 선택은 TEST에서 시장 선택보다 hit rate와 selected-event Brier가 뒤졌다는 반증.
2. anchored conditional logit/Base Margin/ensemble은 reused TEST에서 DeltaLL CI가 양수였다는 확률정보 신호.

확률합 1.0 검증도 포함한다. source: `data_model_evidence.json/rank_and_probability_metrics|market_baseline_comparison`, `../revised_v11_seoul_bugyeong_full_rerun_20260822/probability_sum_validation.csv`, `../revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/delta_ll_metrics.csv`.

### `#bets` — 10. 공식 환급 7개 승식 수익성

두 패널로 분리한다.

- v11 full-rank VALID-locked 정책 7승식.
- v11 market-anchor VALID-locked 정책 7승식.

각 표에 model, policy, bets, hits, hit rate, ROI, daily-block CI, 평균 선택배당/평균 적중환급, MDD, Sharpe, Kelly status/return, 시장 ROI 또는 EDGE correlation을 가능한 범위에서 둔다. 빈 칸 대신 `미측정/식별 불가`를 쓴다.

필수 판정: 두 계열 모두 확인된 수익 승식 0/7. anchor 복승 +39.72%의 CI 하한 -3.30%, full-rank 연승 +1.26%의 CI 하한 -26.41%를 함께 보여 점 ROI만으로 승격하지 않는 사례로 쓴다.

source: `betting_edge_evidence.json/v11_locked_full_rank_seven_bets|v11_market_anchor` 및 아래 chart 원천.

### `#darkhorse` — 11. 8~15배, 상위 10~40%, 이변 피처

세 하위 섹션:

1. v11 8~15배 재검증: 94건, ROI +9.79%, CI -46.31%~+74.21%, 과거 +32.69% 미재현.
2. H11C VALID top10/20/30/40: 네 cutoff 모두 H11B가 equal-count q control보다 뒤졌고 Holm 0/4.
3. upset grid: 56셀 중 point ROI≥30% 22, CI-positive 0, FDR 0. 가장 큰 complex-bet ROI와 최대 적중 제거 민감도를 나란히 둔다.

0.5% 중요도 문턱은 운영 문턱이지 통계적 유의수준이 아님을 반복한다. 29피처 전체 순위는 접는 표, CI가 양수인 `tr_plcrate`, `hr_rest_days`, `wgBudam_chg`, `hr_plcrate` 네 개는 펼친 표로 둔다. `clinic_30d`는 점중요도 1위지만 H7 ablation 미확인이라는 충돌을 같은 카드에서 설명한다.

source: `betting_edge_evidence.json/v11_upset_feature_and_top_fraction|odds_8_15_evidence|h5_to_h9.h7_clinic_ablation`.

### `#h5-h18` — 12. H5~H18 가설 원장

한 행이 하나의 가설이 되도록 다음 필드를 표로 고정한다.

`ID / 가설 / 실제 fit·validation 수 / 평가구간 / 주확률지표 / ROI·시장차이 / multiplicity / 증거상태 / stake / 현재판정`

최소 포함:

- H5 adaptive/fixed rolling: paired 손실방어는 통과했으나 절대 ROI 미확인.
- H6 calibration: proper score 일부 개선, Holm 실패, Kelly 악화.
- H7 clinic ablation: 증분 미확인.
- H8 later-date fresh64: 확률신호, optional continuation.
- H9 fresh64 공식 7승식: 0/7.
- H10A M6: 재현 가능하지만 M5 대비 악화, fresh64 7승식 0/7.
- H11B: 실제 다중모델 학습과 prospective freeze.
- H11C: VALID 문턱 잠금, Holm 0/4.
- H11D: shrinkage 미확인.
- H11E: 29피처 중 23개 provenance 미확보로 차단.
- H12/H15/H16/H18: synthetic/readiness PASS이지 ROI 증거 아님.
- H13/H14: pool weight 불안정, late28 재사용.
- H17: ridge가 약 98.08% 시장으로 수축, 작은 DeltaLL 미확인.

source: `betting_edge_evidence.json/h5_to_h9|h10_to_h18`, 세부 학습증거는 `data_model_evidence.json/actual_training_inventory`.

### `#fresh` — 13. later-date fresh64와 prospective 미정산

fresh64 5피처와 M6를 분리한다. 5피처 공식 7승식의 쌍승 +0.94%만 양수였지만 CI -82.95%~+94.78%, Holm paired 0/7임을 보인다. M6는 7승식 모두 음수, paired 0/7이다. fresh64가 pristine이 아닌 이유(initial27 뒤 fresh37)를 표 제목에 포함한다.

그 아래 prospective snapshot 카드:

- 고정시각을 반드시 표시.
- 17/17 `WAITING_T5_WIN_PLACE`.
- H13/H14/H16/H17/H18 frozen race 0.
- outcome/result/dividend rows 0, network settlement calls 0, live stake 0.

이 카드는 자동으로 최신 상태처럼 표현하지 않는다. `인벤토리 고정시각 현재`라고 쓴다. source: `betting_edge_evidence.json/h5_to_h9.h8_later_date_fresh64|h9_fresh64_official_seven_bets|h10_to_h18.h10a_six_feature_fresh64_seven_bets|prospective_20260823_snapshot_at_inventory`.

### `#economics` — 14. EV·공제·가격·Kelly·slippage

다음 게이트를 순서도 형태로 표시한다.

`확률 p → 실행시점 조합배당 O → EV pO-1 → 공제 반영 여부 → 슬리피지/풀충격 → Kelly/cap → OOS bankroll`

현재 측정상태:

- 공식 최종환급에는 공제가 반영되어 20%/27% 이중차감 금지.
- 안전한 T-5 full-grid 7승식 조합배당 미검증.
- closing price는 executable quote가 아님.
- 복합승식 실제 EV/Kelly는 losing combination의 사전가격 부재로 식별 불가.
- deep 8~15 연승은 haircut 0/2/5/10/20%에서 ROI -0.37/-2.36/-5.35/-10.33/-20.29%.

source: `betting_edge_evidence.json/price_fee_kelly_and_slippage_limitations|deep_and_stack_defensive_evidence.slippage_haircut_sensitivity`.

### `#venues` — 15. 서울·부경 안정성

서울·부경의 데이터 포함 여부, 행/경주 수, rank metric, anchor DeltaLL을 보인다. 경마장별 결과가 모두 동일하다고 요약하지 않는다. venue-specific ROI의 CI-positive 셀이 없다는 기존 v11 결론을 유지하고, 최종 다이제스트에 없는 추가 venue 숫자는 만들지 않는다.

source CSV:

- `../revised_v11_seoul_bugyeong_full_rerun_20260822/venue_stratified/venue_rank_metrics.csv`.
- `../revised_v11_seoul_bugyeong_full_rerun_20260822/venue_stratified/venue_anchor_metrics.csv`.
- `../revised_v11_seoul_bugyeong_full_rerun_20260822/venue_stratified/venue_betting_metrics.csv`.

### `#xai` — 16. 왜 EDGE가 생겼는가와 해석 한계

전역 중요도와 국소 사유를 분리한다. upset permutation importance 29개, CI-positive 4개, clinic ablation 반증을 우선한다. 과거 v11의 `top_upset_horse_local_reasons.csv`는 reused TEST 사례 설명으로만 사용하고, 특정 말에 대한 실전 추천으로 표현하지 않는다. SHAP이 없는 모델을 SHAP으로 표현하지 말고 해당 산출물의 실제 방법(permutation/tree/local reason)을 적는다.

source:

- `../revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/upset_feature_importance.csv`.
- `../revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/top_upset_horse_local_reasons.csv`.
- `../clinic30d_rolling_ablation_20260823/experiment_summary.json`.

### `#failures` — 17. 실패, 충돌, 오래된 결과

양의 결과보다 먼저 다음을 표로 고정한다.

- v7 137피처 final-pool 누수와 서울-only 한계.
- 동일데이터에서 고른 historical 8~15 top5 +47.47%는 OOS 아님.
- Benter 과거 +32.69% ROI CI가 0 포함, v11에서 같은 크기로 미재현.
- fresh64 optional continuation.
- H11B 최초 중단 fit 및 이후 성공 fit을 모두 보존.
- H11D whole-file hash의 물리 TEST-byte read.
- H13/H14 weight instability와 H17 market shrinkage.
- H11E feature provenance blocker.
- ROI 30% 초과 셀의 누수/선택편향/희귀적중 경고.
- one-day settlement도 장기 성과 확증이 아님.

source: `data_model_evidence.json/conflicts_stale_and_failed_results|honest_limits`, `betting_edge_evidence.json/historical_superseded_or_nonconfirmatory|failures_and_limits`.

바로 뒤에 `chronology_literature_evidence.json/metrics_not_measured`의 16개 항목을 `metric / status / reason` 표로 펼쳐 둔다. `NOT_MEASURED`, `NOT_REACHED`, `PARTIALLY_UNAVAILABLE`, `NOT_IDENTIFIABLE`, `NOT_AVAILABLE_IN_SAVED_SOURCES`, `NOT_DETERMINED`를 서로 다른 상태로 유지한다. 특히 2026-08-23 확률·Top3·7승식 성과, 실제 slippage/latency/odds drift, pool impact/capacity, 리베이트 분해, 장기 pristine ROI, H13~H17 통합 다중검정, 인과효과, 실제 조합시장 calibration, 독립감사 회사 ROI, 자동주문 법률 가능성은 측정되지 않았다.

### `#synthesis` — 18. 시장확률보다 나은 베팅이 가능한가

결론은 세 층으로 쓴다.

1. **확률층:** 일부 앵커/rolling 분석에서 시장 대비 양의 DeltaLL이 반복됨.
2. **선택층:** 8~15 연승에서 재사용 OOS 상대 손실방어가 가장 강하지만 H11C VALID 결과와 FDR이 일관되게 지지하지 않음.
3. **실행층:** 실행시점 조합가격, 완전 신규 표본, 슬리피지/풀충격이 없어 수익 배포 불가.

허용되는 연구 결론은 `제한적 시장상대 확률정보와 방어 가능성`, 금지되는 결론은 `시장보다 나은 실전 베팅법 확립`이다.

### `#protocol` — 19. 다음 검증 프로토콜

새 실험 결과를 쓰지 말고 이미 잠긴 후속 조건만 정리한다.

- 완전 신규 forward OOS.
- T-5 또는 실제 의사결정 시점 7승식 조합가격의 결과분리 안전 소스.
- 사전등록 모델/문턱/티켓/다중검정군.
- 공식환급에서 공제 이중차감 금지.
- payout haircut 및 capacity/pool-impact 민감도.
- 최소 기간·경주수 도달 전 ROI/MDD/Sharpe 확증 금지.
- stake 0 유지, go/no-go 조건을 수치로 잠그기.

### `#sources` — 20. 근거 파일과 외부 출처

내부 근거는 세 다이제스트의 manifest/index를 합쳐 `ID / 상대경로 / SHA-256 / bytes / 사용 섹션` 표로 만든다. 중복 파일은 SHA와 상대경로가 같을 때 한 번만 표시한다. 외부 출처는 chronology/literature 다이제스트에 실제로 수록된 자료만 `문헌 ID / 제목 / 유형 / URL / 확인일 / 사용 주장`으로 둔다.

### `#conclusion` — 21. 최종 결론

맨 마지막에는 다음 판정을 짧게 반복한다.

- 데이터·실제 학습 증거: 확인.
- 일부 확률정보 우위: 재사용/retrospective 범위에서 관찰.
- 7승식 확정 절대수익: 0.
- prospective 성과: 미정산.
- 배포: NO-GO, stake 0.

## 3. 차트 구현 목록

모든 chart는 inline SVG, 명시적 0선, 단위 표시, `<title>`, `role="img"`, `aria-label`, source/field를 적은 `<figcaption>`을 가져야 한다. CI가 있으면 점만 그리지 말고 whisker를 그린다. CI가 없는 값은 별도 모양과 `CI 미측정` 표기를 사용한다.

### C01. 증거 사다리

- 형태: 6단계 수직 flow.
- source: `betting_edge_evidence.json/evidence_scope_taxonomy`.
- fields: `scope`, `meaning`.
- 경고: 순위는 설명용이며 정량 점수가 아니다.

### C02. 데이터 분할 타임라인

- 형태: train/valid/test 기간 막대.
- source: `data_model_evidence.json/canonical_dataset.split_files`.
- fields: `split`, `date_min`, `date_max`, `rows`, `races`.
- 표기: test는 `재사용 TEST` amber badge.

### C03. 서울·부경 분할 커버리지

- 형태: split별 stacked rows 또는 races.
- source: 같은 `split_files`.
- fields: `meet_rows.서울`, `meet_rows.부경`, `meet_races.서울`, `meet_races.부경`.
- 주의: rows와 races를 한 축에 혼합하지 않는다.

### C04. 전처리 감사 패널

- 형태: 검사 항목 PASS 표/아이콘; 성능 chart가 아님.
- source: `data_model_evidence.json/canonical_dataset.preprocessing_integrity`.
- fields: `nan_model_values_by_split`, `infinite_model_values_by_split`, `correlation.pairs_before_pruning`, `pairs_after_pruning`, `direct_recheck_max_absolute_pairwise_correlation`, `strict_target_encoding.checks_passed`, `checks_total`.
- 경고: Pearson pruning PASS는 비선형 중복 부재를 뜻하지 않는다.

### C05. 6모델 TEST NDCG@5

- 형태: 가로 막대, 4자리 소수.
- source: `../revised_v11_seoul_bugyeong_full_rerun_20260822/model_metric_comparison.csv`.
- fields: `model`, `test_ndcg_at_5`.
- 정렬: 값 내림차순. 유의차 검정이 없으므로 우승색 과장 금지.

### C06. 순위지표별 우승자 비교

- 형태: small multiples 또는 정규화하지 않은 4개 패널.
- source: 같은 CSV.
- fields: `test_mean_spearman`, `test_mean_absolute_rank_error`, `test_top1_hit_rate`, `test_top3_ordered_exact_match_rate`, `test_top3_unordered_exact_match_rate`.
- 주의: MAE만 낮을수록 좋고 exact 분모는 1,078경주다.

### C07. 시장 앵커 DeltaLL forest

- 형태: 점+daily-block 95% CI, 0선.
- source: `../revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/delta_ll_metrics.csv`.
- filter: `split=test`.
- fields: `model`, `delta_ll_per_race`, `delta_ll_ci_low_daily_block`, `delta_ll_ci_high_daily_block`, `races`, `delta_ll_confirmed_positive_95pct`.
- badge: `reused TEST probability evidence`.

### C08. full-rank 7승식 ROI forest

- 형태: 승식별 ROI 점+CI, 시장 ROI를 작은 보조점으로 overlay.
- source: `../revised_v11_seoul_bugyeong_full_rerun_20260822/locked_policy_test_results.csv`.
- fields: `bet_type`, `test_roi`, `test_roi_ci_low_daily_block`, `test_roi_ci_high_daily_block`, `market_roi`, `test_bets`, `test_wins`, `profit_confirmed_95pct`.
- 색: point ROI 양수라도 CI 미통과면 amber, green 금지.

### C09. market-anchor 7승식 ROI forest

- source: `../revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/locked_test_results.csv`.
- fields: `bet_type`, `test_roi`, `test_roi_ci_low_daily_block`, `test_roi_ci_high_daily_block`, `test_mean_predicted_edge`, `test_edge_spearman_with_realized_edge`, `test_bets`, `test_wins`.
- 주의: full-rank와 동일 x축 범위를 쓰거나, 범위 차이를 명시한다.

### C10. 이변 피처 중요도

- 형태: 상위 15 가로 막대+CI, weight 0.005 수직선.
- source: `../revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/upset_feature_importance.csv`.
- fields: `rank`, `feature`, `positive_importance_weight`, `ci_low_daily_block`, `ci_high_daily_block`, `weight_threshold`, `selected_for_retraining`, `ci_confirmed_positive`.
- 경고: 막대 weight와 permutation drop CI는 동일 단위인지 원천 정의를 따라 각각 라벨; 임의로 같은 축에 겹치지 않는다.

### C11. top10~40 upset ROI heatmap

- 형태: `model_key × fraction_label` 행, `bet_type` 열, 색은 `roi`.
- source: `../revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/upset_bet_summary_independently_verified.csv`.
- fields: `model_key`, `fraction_label`, `bet_type`, `roi`, `roi_ci_low_daily_block`, `roi_ci_high_daily_block`, `roi_fdr_q_value`, `profit_confirmed_fdr_5pct`, `bets`, `wins`.
- overlay: FDR 통과 셀만 별표. 현재 별표는 0개여야 한다.

### C12. 최대 적중 제거 민감도

- 형태: point ROI와 `roi_without_largest_winning_return` dumbbell.
- source: C11과 같은 CSV.
- fields: `model_key`, `fraction_label`, `bet_type`, `roi`, `roi_without_largest_winning_return`, `largest_winning_dividend`, `largest_winner_share_of_winning_gross`, `bets`, `wins`.
- filter: 상위 point ROI 셀을 보여주되 선택 규칙을 caption에 고정.

### C13. 8~15배 결과 forest

- 형태: v11 revalidation, H5 adaptive, H5 fixed, deep blend를 분리된 행으로 점+CI.
- sources/fields:
  - `../odds_8_15_v11_revalidation_20260823/odds_8_15_policy_results.csv`: `model_key`, `policy`, `split`, `bets`, `roi`, `roi_ci_low`, `roi_ci_high`.
  - `../rolling_origin_market_challenger_20260823/fold_place_policy_results.csv`: 집계행이 없다면 다이제스트 `h5_to_h9.h5_adaptive_rolling`의 `place_model_roi`, `place_roi_ci_low/high`, `same_race_market_roi`, `paired_advantage`, `paired_ci_low/high`를 사용.
  - `../fixed_lock_rolling_stability_20260823/fold_place_metrics.csv`: 동일하게 다이제스트 `h5_fixed_valid_lock` 집계값 사용.
  - `betting_edge_evidence.json/deep_and_stack_defensive_evidence.eight_to_fifteen_place_cell`.
- 색/shape로 `absolute ROI`와 `paired model−market`를 분리한다.

### C14. H11C VALID cutoff 비교

- 형태: top10/20/30/40별 H11B와 q-control ROI grouped bar+CI.
- source: `../h11c_darkhorse_8_15_preregistered_20260823/valid_cutoff_metrics.csv`와 `valid_paired_metrics.csv`.
- cutoff fields: `cutoff`, `strategy`, `tickets`, `roi`, `daily_block_ci_low`, `daily_block_ci_high`, `mdd_percent_total_stake`, `ece10_model_probability`.
- paired source `../h11c_darkhorse_8_15_preregistered_20260823/valid_paired_metrics.csv`; fields: `cutoff`, `quantile`, `edge_threshold`, `paired_roi_difference`, `daily_block_ci_low`, `daily_block_ci_high`, `one_sided_p`, `test_kind`, `sign_patterns`, `day_clusters`, `holm_adjusted_p`, `holm_pass_0_05`.
- badge: `VALID selection only`, 성과 확증으로 표현 금지.

### C15. fresh64 5피처 공식 7승식

- 형태: ROI forest; 보조 tooltip/table에 hit, average dividend, MDD, Sharpe, EDGE.
- source: `../fresh_all_bets_extended_20260823/fresh_all_bet_summary.csv`.
- fields: `bet_type`, `bets`, `hits`, `hit_rate`, `roi`, `roi_daily_block_ci_low`, `roi_daily_block_ci_high`, `mean_final_dividend_among_hits`, `mdd_fraction_of_100_unit_initial_bankroll`, `sqrt_n_sharpe_per_bet_not_annualized`, `mean_probability_edge`, `deployment_eligible`.
- badge: `later-date optional continuation`, `0/7 confirmed`.

### C16. fresh64 paired 시장우위

- 형태: paired ROI difference forest.
- source: `../fresh_all_bets_paired_advantage_extended_20260823/paired_advantage_summary.csv`.
- fields: `bet_type`, `paired_roi_difference`, `paired_daily_block_ci_low`, `paired_daily_block_ci_high`, `exact_day_signflip_p_one_sided`, `holm_adjusted_exact_signflip_p`, `holm_5pct_paired_advantage`, `comparator_kind`.
- 현재 Holm 별표 0개를 검증한다.

### C17. M5와 M6 fresh64 비교

- 형태: 승식별 ROI dumbbell 또는 grouped bar.
- sources: `../fresh_all_bets_extended_20260823/fresh_all_bet_summary.csv`와 `../h10a_provenance_six_feature_all_bets_posthoc_20260823/fresh_all_bet_summary.csv`.
- join: `bet_type`만. fields: `roi`, `roi_daily_block_ci_low/high`, `bets`, `hits`.
- 경고: 둘 다 관찰된 optional continuation이며 M6는 post hoc.

### C18. H11B 모델별 DeltaLL

- 형태: VALID와 reused TEST facet forest.
- source: `../h11b_six_feature_multimodel_preregistered_20260823/historical_probability_metrics.csv`.
- fields: `split`, `model_key`, `races`, `delta_ll_per_race`, `delta_ll_daily_block_ci_low`, `delta_ll_daily_block_ci_high`, `brier_improvement`, `top1_hit_rate`, `maximum_probability_sum_error`.
- target-date 행은 없으며 만들지 않는다.

### C19. H13/H14/H17 regularization 결과

- 형태: 세 패널 proper score 및 별도 weight/exponent annotation.
- sources:
  - `../h13_market_anchor_nonnegative_log_pool_preregistered_20260823/late28_metrics.csv`.
  - `../h14_convex_market_residual_pool_preregistered_20260823/late28_metrics.csv`.
  - `../h17_ridge_convex_market_residual_pool_preregistered_20260823/late28_metrics.csv`.
- common fields: `model`, `winner_conditional_nll`, `runner_brier`, `ece10_equal_width`, `race_top1_hit_rate`, `maximum_race_probability_sum_error`.
- weights/exponent: `betting_edge_evidence.json/h10_to_h18.h13_nonnegative_log_pool|h14_convex_pool|h17_ridge_convex_pool`.
- badge: `reused late-VALID`, CI가 없는 proper-score 점을 확정 개선으로 표현 금지.

### C20. payout haircut 민감도

- 형태: x=haircut, y=ROI line, 0선.
- source: `../deep_market_challengers_20260823_v2/independent_audit/payout_slippage_sensitivity.csv`.
- fields: `audit_cell`, `bet_type`, `model_key`, `payout_haircut`, `bets`, `model_roi`, `market_roi`, `paired_roi_difference`.
- 기본 facet: 8~15 연승 deep blend. 다른 audit cell을 자동 혼합하지 않는다.

### C21. 서울·부경 anchor DeltaLL

- 형태: venue×model forest.
- source: `../revised_v11_seoul_bugyeong_full_rerun_20260822/venue_stratified/venue_anchor_metrics.csv`.
- fields: `venue`, `model_key`, `races`, `delta_ll_per_race`, `delta_ll_ci_low_daily_block`, `delta_ll_ci_high_daily_block`, `model_top1_hit_rate`, `market_top1_hit_rate`, `brier_improvement_vs_market`.
- 경고: reused TEST venue stratification.

### C22. 연구 연대기·증거상태

- 형태: 시간축 card/timeline; 성과 크기 chart가 아님.
- source: `chronology_literature_evidence.json/chronology`의 15개 레코드.
- fields: `order`, `period`, `phase`, `question`, `work`, `result`, `scientific_status`, `confidence`, `evidence`.
- 상태색: superseded=gray, failed=red, retrospective=amber, prospective-unsettled=purple, validated implementation=blue.

## 4. 표 구현 목록

차트가 요약을 담당하고 표가 정확한 값을 보존한다. 최소 표:

1. 용어·산식.
2. 증거상태 정의.
3. 문헌→프로젝트 적용 매핑.
4. 현업 수익원과 자료유형.
5. 프로젝트 연대기.
6. split/venue 데이터 구성.
7. 전처리·누수 감사.
8. Git-v11 비교와 비동형 한계.
9. 실제 6모델 학습·저장·로그.
10. 6모델 rank metrics.
11. full-rank 대 시장 hit/Brier 반증.
12. anchor DeltaLL.
13. full-rank 7승식.
14. anchor 7승식.
15. 29 upset feature와 CI-positive 4개.
16. top10~40/FDR·largest-hit 요약.
17. 8~15 결과 계보.
18. H5~H18 가설 원장.
19. fresh64 5피처·M6 7승식.
20. prospective readiness와 고정시각.
21. EV/Kelly/가격 가용성 매트릭스.
22. slippage sensitivity.
23. venue 안정성.
24. 실패·충돌·superseded 결과.
25. 내부 SHA 근거색인.
26. 외부 출처 목록.

세부 피처 29개, 전체 H5~H18 파일목록, SHA index는 `<details>`로 접는다. 핵심 7승식·fresh64·최종 한계는 접지 않는다.

## 5. 필수 경고문

다음 문구는 의미를 약화하지 말고 해당 섹션 가까이에 반복한다.

### W1. 확률과 수익

> 양의 DeltaLL 또는 양의 EDGE는 양의 EV·ROI를 뜻하지 않는다. 실행 가능한 티켓 가격, 공제, 슬리피지, 선택 규칙, 불확실성 검증이 별도로 필요하다.

### W2. 표본 재사용

> v11 TEST와 여러 late-VALID 구간은 프로젝트에서 이미 관찰됐다. 이 결과는 시간순 강건성 근거일 수 있으나 완전 신규 독립 확증이 아니다.

### W3. fresh64

> fresh64는 later-date지만 initial27을 본 뒤 fresh37을 추가한 optional continuation이다. pristine preregistered holdout으로 부르지 않는다.

### W4. 마감가격

> closing/result odds는 T-5 실행가격으로 검증되지 않았다. 마감배당 기반 EDGE와 ROI는 실제 체결가능성을 보장하지 않는다.

### W5. 복합승식 EV/Kelly

> losing combination의 사전 조합가격이 없으므로 복합승식의 실제 EV와 Kelly는 식별 불가다. 적중 티켓의 최종환급만으로 대체하지 않는다.

### W6. 공제

> 공식 최종환급에는 KRA 공제가 이미 반영되어 있다. 단승 20%, 다중승식 27%를 다시 차감하지 않는다.

### W7. 희귀 적중·다중검정

> 상위 이변/복합승식의 큰 점 ROI는 소수 적중에 집중됐다. 56개 upset cell 중 CI-positive와 FDR 통과는 모두 0개다.

### W8. prospective

> 인벤토리 고정시각 현재 target outcome·result·dividend 관측은 0이다. validator PASS는 수익성 PASS가 아니다.

### W9. 데이터 무결성의 범위

> 숫자형·결측·상관·시간분할 감사 PASS는 모든 비선형 중복, 모든 도메인 누수, 미래 운영 drift의 부재를 증명하지 않는다.

### W10. 배포

> 현재 배포 판단은 NO-GO, 실베팅 stake 0이다.

## 6. 근거 링크 규칙

### 6.1 로컬 파일

- 모든 href는 최종 `index.html` 기준 상대경로와 forward slash를 사용한다. `file:///`, 절대 Windows 경로, 백슬래시를 금지한다.
- 최종화 다이제스트: `evidence/betting_edge_evidence.json`, `evidence/data_model_evidence.json`, `evidence/chronology_literature_evidence.json`.
- sibling report: `../<report_folder>/<file>`.
- data/model/source: 최종 보고서 폴더에서 실제 상대경로를 계산해 사용하며 문자열을 손으로 추측하지 않는다.
- 링크 텍스트는 `[B01]`, `[D07]`, `[L03]`처럼 namespace를 나눠 충돌을 피한다. 각 ID는 title/aria-label에 정확한 상대경로와 SHA 앞 12자를 포함한다.
- 숫자가 있는 표/차트마다 최소 한 개의 local source link를 바로 아래에 둔다. 여러 파일을 join했으면 모두 링크한다.
- `figcaption` 예: `근거 [B04]: locked_policy_test_results.csv · fields: bet_type, test_roi, test_roi_ci_low_daily_block, test_roi_ci_high_daily_block · scope: reused TEST`.
- SHA는 다이제스트 manifest/index에 있는 값과 현재 파일 hash가 일치할 때만 표시한다. 불일치 시 빌드를 실패시킨다.

### 6.2 외부 자료

- chronology/literature 다이제스트의 `literature_and_methodology.selected_research`, `industry_and_company_methods`, `legal_boundary`에 실제 포함된 URL만 사용한다.
- `<a target="_blank" rel="noreferrer">`.
- 링크 근처에 저장 카탈로그가 제공하는 자료유형(`peer-reviewed`, `preprint`, `regulator`, `company statement`, `SEC/annual report`)을 표시한다. 이번 finalization은 원문을 다시 조회하지 않았으므로 새 확인일을 만들지 않고 `저장 출처 재사용, 최신성 미갱신` 경고를 표시한다.
- 회사 주장과 독립감사 결과를 같은 badge로 표시하지 않는다.
- 인용문은 최소화하고, 우리 프로젝트에 적용한 해석은 `해석`으로 명시한다.
- DOI/공식기관/공시 원문을 우선한다. search-result URL은 금지한다.

### 6.3 수치의 단위와 표시

- JSON의 비율 0.0126은 HTML에서 `+1.26%`로 표시할 수 있으나 원본값을 `data-raw-value` 또는 snapshot에 남긴다.
- DeltaLL은 퍼센트로 바꾸지 않고 6자리 소수.
- CI는 점추정치와 같은 단위.
- `null`, 빈 문자열, `unavailable_*`을 0으로 렌더링하지 않는다.
- average odds와 average winning dividend를 혼용하지 않는다.
- MDD는 source column의 기준(100-unit bankroll 또는 total stake)을 caption에 쓴다.
- Sharpe는 `비연율화 sqrt(n) Sharpe` 또는 원본 정의를 정확히 표시한다.

## 7. 빌드 산출물과 snapshot

생성기가 만들어야 할 파일:

- `index.html`: self-contained HTML/CSS/inline SVG.
- `report_data_snapshot.json`: 세 digest의 현재 SHA-256, 사용한 JSON path, chart별 source/field/filter, 렌더링된 핵심 수치.
- `report_manifest.json`: HTML bytes/SHA, 생성시각 KST, 입력 6개 파일과 핵심 원천 hash, overall verdict.
- `html_validation.json`: 아래 정적 검증 결과.
- 선택적 `README.txt`: 열람 방법과 NO-GO 한 줄. 단, 프로젝트와 맞지 않는 설명을 새로 만들지 않는다.

PNG를 생성하지 않아도 된다. PDF가 후속으로 필요하면 같은 HTML을 print 변환하되 HTML과 PDF의 KPI·표 수치 hash/snapshot이 같아야 한다.

## 8. 정적·시각 검증 계약

최소 자동검사:

1. HTML5 doctype, UTF-8, 정확히 H1 하나.
2. 모든 필수 section id 22개 존재.
3. 내부 link와 navigation anchor 100% resolve.
4. 세 JSON parse PASS, 세 MD 존재.
5. 모든 manifest/index 원천 존재 및 SHA 일치.
6. unresolved template token 0개.
7. `0 confirmed profitable`, `NO-GO`, `stake 0`, `prospective unsettled` 문구 존재.
8. 7승식 이름 단승·연승·복승·쌍승·삼복승·삼쌍승·복연승 모두 존재.
9. `historical/reused TEST`, `optional continuation fresh64`, `valid selection only`, `synthetic readiness` 경계 모두 존재.
10. `EV 식별 불가`, `closing odds not executable`, `공제 이중차감 금지` 존재.
11. 29 selected feature, CI-positive 4, upset FDR 0/56 존재.
12. actual full-rank models 6, 203/203 validation, probability sum 오차 존재.
13. prospective snapshot의 as-of 시각, waiting 17, outcome rows 0, stake 0 존재.
14. 최소 20 evidence table, 최소 12 inline SVG. 구현상 필요한 chart만 남길 수 있으나 C07~C20 중 핵심 10개는 필수다.
15. 모든 SVG에 title/aria-label, 모든 table에 header scope.
16. mobile 360px에서 가로 overflow는 `.table-wrap` 내부만 허용.
17. print preview에서 차트/표가 잘리지 않고 링크 URL이 본문을 덮지 않음.
18. dark/light 대비 WCAG AA를 목표로 확인.
19. 외부 URL은 HTTP(S), local link는 상대경로.
20. 기존 v11 report의 stale 문구나 현재 확증과 충돌하는 `수익 확정 전략이 있다` 같은 문장 부재.

시각 QA를 실행하지 못하면 `browser visual QA not performed`로 남기고 PASS로 위장하지 않는다.

## 9. 생성기 구현 순서

1. 입력 6개 존재·parse 검사.
2. 다이제스트와 각 manifest의 SHA 재검산.
3. evidence adapter를 만들어 숫자와 scope를 동시에 읽음.
4. 핵심 판정 invariants를 assert: confirmed deployable profit 0, stake 0, prospective result rows 0.
5. table data와 chart data를 같은 in-memory record에서 생성해 불일치 방지.
6. inline SVG 생성 시 0선·CI·단위·scope caption 포함.
7. report_data_snapshot 작성.
8. HTML 작성.
9. 상대링크/anchor/필수문구/수치 invariant 정적 검증.
10. 가능하면 실제 브라우저로 desktop/mobile/dark/print 시각검증.
11. HTML SHA를 계산해 manifest에 기록.

어떤 단계에서 원천이 없거나 필드가 바뀌면 숫자를 추측해 채우지 말고 해당 chart/table을 `SOURCE_SCHEMA_MISMATCH`로 빌드 실패시킨다. 정말로 미측정인 항목은 실패시키지 않고 `미측정/식별 불가/미정산`으로 명시한다.

## 10. 최종 보고서의 금지 표현과 허용 표현

| 금지 | 허용 |
|---|---|
| “수익 전략을 찾았다” | “확정된 절대수익 전략은 없다” |
| “fresh64 독립 확증” | “later-date optional continuation” |
| “모델이 시장을 이겼다” | “일부 reused OOS에서 DeltaLL 또는 paired 손실방어가 관찰됐다” |
| “29개 유의 피처” | “0.5% 운영 문턱 통과 29개, CI-positive 4개” |
| “Kelly로 수익 극대화 가능” | “실행가격이 있는 티켓에서만 Kelly 식별 가능” |
| “모든 전처리 완벽” | “기록된 숫자형·결측·상관·시간안전 감사 범위에서 PASS” |
| “validator PASS이므로 당일 수익 가능” | “구현 준비 PASS, target 성과 미정산” |
| “closing odds로 실전 EV 확인” | “closing odds와 실행가격 차이가 미해결” |

## 11. 보고서 마지막 한 문장

최종 HTML은 다음 의미로 끝나야 한다.

> 이 연구는 시장이 완전히 효율적이지 않을 가능성을 확률지표에서 일부 포착했지만, 실행 가능한 사전가격과 독립 forward OOS에서 수수료·슬리피지·다중검정을 이긴 절대수익을 아직 확인하지 못했다. 현재 단계의 정직한 성과는 수익 시스템이 아니라 재현 가능한 데이터·학습·평가 기반과 제한적인 시장상대 손실방어 가설이며, 배포는 NO-GO이고 실베팅액은 0이다.
