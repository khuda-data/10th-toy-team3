# 최종 종합 연구보고서용 연대기·문헌·검증 근거 인벤토리

작성 시점: 2026-08-23 10:25 KST  
상태: `FINALIZATION_EVIDENCE_ONLY`  
정본 데이터: `chronology_literature_evidence.json`

## 1. 범위와 안전 경계

이 문서는 이미 로컬에 저장된 데이터, 로그, 보고서, 사전등록, 독립 검증 결과만 다시 정리한 근거 인벤토리다. 이번 작업에서는 신규 웹 검색, 신규 학습, 신규 백테스트, 신규 실험, 사후 production 실행을 하지 않았다. 2026-08-23 목표일의 outcome, result, dividend, settlement 자료를 읽지 않았고, 네트워크 호출·신규 모델 적합·신규 백테스트·실제 stake는 모두 0이다.

신뢰도 표기는 다음과 같다.

- `HIGH`: 로컬 원본 산출물·해시·독립 validator로 직접 결속됨.
- `MEDIUM`: 기존 보고서에 저장된 외부 문헌·회사 자료를 재사용했으나 이번 작업에서 원문·최신성을 다시 조회하지 않음.
- `LOW`: 간접 추론뿐인 경우. 이 인벤토리의 핵심 판정에는 사용하지 않았다.

최종 증거 경계는 명확하다. 프로젝트는 재사용 역사 표본에서 시장확률에 추가되는 작은 확률정보와 일부 방어적 paired 우위를 관측했고, 엄격한 전처리·재현·사전동결 인프라를 구축했다. 그러나 독립적인 장기 전향 표본에서 수수료·실행가격·시장충격 후 절대수익을 확인하지 못했다. 따라서 현재 판정은 **연구 인프라 PASS, 실전 배치 NO-GO, stake 0**이다. 신뢰도 `HIGH`.

핵심 근거:

- `Report_summary.txt`
- `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/final_validation.json`
- `outputs/reports/market_probability_advantage_deep_research_20260823/report_data_snapshot.json`
- `outputs/reports/20260823_prospective_master_registry/master_registry.json`
- `outputs/reports/20260823_prospective_master_registry/independent_validation.json`

## 2. 실제 작업 연대기

| 순서 | 기간 | Phase | 실제 결과와 과학적 판정 | 핵심 근거 | 신뢰도 |
|---:|---|---|---|---|---|
| 1 | 2026-08-19~20 | 초기 전처리·전체 착순 모델 v5~v9 | 여러 수치화·스케일링 버전과 RF/XGB/LGBM/CatBoost/Deep RankNet/PL 계열을 실제 학습했다. 순위지표는 계산됐지만 validation 잠금 정책에서 95% CI 하한이 양수인 수익정책은 없었다. `MECHANICAL_SUCCESS_ECONOMIC_NOT_CONFIRMED`. | `Report_summary.txt`; `outputs/reports/project_learning_textbook_20260823/index.html` | HIGH |
| 2 | 2026-08-20~21 | Benter형 시장 앵커 전환 | `log(q)`를 고정 offset으로 두고 조건부 로짓, 2단계 결합, Base Margin boosting, walk-forward를 구현했다. 역사 표본의 양의 ΔLL은 있었으나 ROI CI는 0을 포함했고 홍콩 외부 재현 ΔLL/race는 -0.0009084, CI [-0.0060925,+0.0039697]였다. | `outputs/reports/benter_market_anchored_20260820/walkforward_summary.json`; `outputs/reports/base_margin_boosting_20260821/summary.json`; `outputs/reports/hk_external_replication_20260821/summary.json` | HIGH |
| 3 | 2026-08-21~22 | 누수·피처·경마장·신규표본 감사 | 2019~2021 축퇴 배당과 일치한 +57.43% 점추정은 배치 근거에서 제외했다. revised_v7은 서울-only였고 서울+부경 기준은 revised_v10 이후 별도로 구성됐다. | `outputs/reports/interim_market_error_research_20260822/meet_scope_audit.json`; `outputs/reports/jeju_bugyeong_validation_20260821`; `Report_summary.txt` | HIGH |
| 4 | 2026-08-22 | revised_v11 서울+부경 clean 재학습 | 56,456행·5,343경주·141개 비시장 피처, TRAIN 33,416/VALID 11,345/TEST 11,695행. 6개 모델 실제 적합·저장·재로딩, 데이터 감사와 203개 검증 PASS. CatBoost TEST NDCG@5 0.778036, Spearman 0.455303, rank MAE 2.439987. 7승식 확정수익 0/7. | `data/revised_v11_seoul_bugyeong_rank_clean_preprocessed/preprocessing_validation.json`; `.../preprocessing_manifest.json`; `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/final_validation.json`; `.../comprehensive_validation.json` | HIGH |
| 5 | 2026-08-22 | v11 시장 앵커·이변·경마장 층화 | 재사용 TEST에서 앵커 ensemble ΔLL/race +0.017194, CI [+0.009507,+0.025014]. 그러나 상위 이변 29피처는 전체 141피처보다 ΔLL이 낮았고 ROI CI/FDR 통과는 0. 서울·부경 각각 양의 앵커 ΔLL은 있었지만 수익 확정은 0. | `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/market_anchor/delta_ll_metrics.csv`; `.../upset_feature/independent_validation.json`; `.../venue_stratified/venue_analysis_validation.json` | HIGH |
| 6 | 2026-08-23 00:00~05:55 | H1~H9 심층 가설·fresh64 공식 정산 | 부분 풀링, attention, log-pool, 0.5% sparse, rolling lock, calibration, clinic ablation, fresh 날짜, 7승식을 실제 실행·감사했다. H5의 재사용 TEST 신호만 제한 지지. H2/H3/H4/H6/H7/H9 미지지. H8 양의 점추정은 optional continuation 한계. fresh64 절대수익 0/7, paired Holm 0/7. | `outputs/reports/market_probability_advantage_deep_research_20260823/report_data_snapshot.json`; `.../index.html`; `outputs/reports/fresh_all_bets_extended_20260823/independent_audit/independent_audit_summary.json` | HIGH |
| 7 | 2026-08-23 06:21~06:28 | H10A 출처완전 6피처 앵커 | `age`, `age__pr`, `oh_sex_암`, `wgBudam_chg`, `wgBudam__pr`, `te_owName`으로 조건부 로짓 6회 적합. 56,456행 변환 replay 최대오차 8.882e-16, 감사 22/22. 재사용 TEST M6-q +0.005422; fresh64 +0.010457, exact p=0.078125. NO-GO. | `outputs/reports/h10a_provenance_six_feature_anchor_20260823/experiment_summary.json`; `.../independent_audit/independent_audit_summary.json` | HIGH |
| 8 | 2026-08-23 06:21~06:59 | H11 V1·H11B 멀티모델 동결 | H11 V1을 주모델로 보존하고 H11B에서 M6/XGBoost/LightGBM/Deep MLP 3 seeds를 실제 학습했다. 총 fit operation 33(최종 성공 15, 이전 중단 18). 25% 동일가중 ensemble VALID ΔLL +0.002420, 재사용 TEST +0.002722로 두 CI 모두 0 포함. 타깃 성과 미측정. | `data/raw_prospective_oos_20260823/preregistration.json`; `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/protocol_before_fit.json`; `.../experiment_summary.json`; `.../independent_validation.json` | HIGH |
| 9 | 2026-08-23 07:00~07:10 | H11C 8~15배 다크호스 VALID 잠금 | H11B EDGE top10/20/30/40 임계값 0.009714/0.007045/0.005444/0.004175를 잠갔다. 네 cutoff 모두 모델 ROI 음수, 동일건수 q 대조 양수, Holm 통과 0. NO-GO. | `outputs/reports/h11c_darkhorse_8_15_preregistered_20260823/threshold_lock.json`; `.../experiment_summary.json`; `.../independent_validation.json` | HIGH |
| 10 | 2026-08-23 07:19~07:39 | H11D 잔차 alpha 스케일링 | VALID 92일을 DEV 64/late 28로 분리. DEV alpha=1.05, late ΔLL +0.002289, CI [-0.003839,+0.008967], p=0.497750. alpha=1 대비 +0.000010245로 미미하고 ECE 악화. deployment alpha=1.06. | `outputs/reports/h11d_market_anchor_shrinkage_preregistered_20260823/protocol_before_analysis.json`; `.../late_valid_metrics.json`; `.../frozen_alpha.json` | HIGH |
| 11 | 2026-08-23 07:43~07:58 | H13 비음수 residual log-pool | SLSQP+analytic gradient로 8회 적합. late28 H13-q +0.002006, CI [-0.004539,+0.009043], p=0.287267; 동일가중 대비 -0.000272. 선택 λ=0, 가중치 합>1 불안정성이 H14 동기가 됨. | `outputs/reports/h13_market_anchor_nonnegative_log_pool_preregistered_20260823/protocol_before_training.json`; `.../experiment_summary.json`; `.../late28_metrics.json` | HIGH |
| 12 | 2026-08-23 07:57~08:20 | H14 convex residual pool | `w>=0`, `sum(w)<=1`로 first44/first64/all92 총 3회 적합. late28 q 대비 +0.002263, CI [-0.003475,+0.008439], p=0.230277; equal 대비 -0.000015, H13 대비 +0.000257, 모두 CI 0 포함. 제약공학 성공, 성능 미확인. | `outputs/reports/h14_convex_market_residual_pool_preregistered_20260823/protocol_before_training.json`; `.../experiment_summary.json`; `.../late28_metrics.json` | HIGH |
| 13 | 2026-08-23 08:21~08:40 | H17 ridge-convex pool | 12 λ first44, 선택 λ first64, all92까지 실제 SLSQP 14회. one-SE로 λ=1. first64/all92 가중치합 0.019227/0.018590로 거의 q에 수축. late28 q 대비 +0.0000837, CI [-0.0000295,+0.0002075], p=0.094291; equal 대비 -0.002195. | `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/protocol_before_training.json`; `.../experiment_summary.json`; `.../late28_metrics.json` | HIGH |
| 14 | 2026-08-23 06:21~10:25 | H11~H18 result-blind T-5 계약 동결 | scorer, 7승식 ticket freezer, probability evaluator, 공식 parser/collector/settler, master registry를 결과 전 동결. registry 159파일, 972/972 PASS, 결과·배당금 read 0, network 0, stake 0. 인프라 성공이지 성과 측정이 아님. | `outputs/reports/20260823_prospective_master_registry/master_registry.json`; `.../independent_validation.json` | HIGH |
| 15 | 2026-08-23 09:07~10:25 | 승인형 사후 fail-closed orchestration | authorization·전체출발·capture·ticket gate, 고정 CLI, append-only retry receipt, 내부파일 재귀 hash, cross-audit release를 구현. cross-audit v2 208/208, metrics 2820/2820, negative 35/35. orchestrator 71/71을 두 번 `REUSE_NO_OVERWRITE`; production 미실행. | `outputs/reports/authorized_postrace_pipeline_preregistered_20260823/protocol_before_first_t5.json`; `.../freeze_complete_cross_audited_before_first_t5.json`; `.../independent_validation_final_cross_audit_v5_frozen.json`; `outputs/reports/h16_h18_settlement_cross_component_audit_20260823/cross_component_audit_v2.json` | HIGH |

## 3. 가설별 성공·실패 판정

| 가설 | 판정 | 근거가 말하는 것 | 근거 파일 |
|---|---|---|---|
| H1 band-specific partial pooling | 제한 지지, 증분 미확인 | q 대비 일부 양의 ΔLL, global 대비 증분 CI 미확정 | `outputs/reports/partial_pooling_market_challenger_20260823/experiment_summary.json` |
| H2 Attention > MLP | 미지지 | 둘 다 양의 역사 점추정이 있었지만 Attention 추가 이점 없음 | `outputs/reports/deep_market_challengers_20260823_v2/experiment_summary.json` |
| H3 residual log-pool 상보성 | 미지지 | VALID가 부분 풀링 가중치를 약 4.5e-8로 제거 | `outputs/reports/market_residual_log_pool_stack_20260823/experiment_summary.json` |
| H4 0.5% 초과 9피처 sparse MLP | 미지지 | sparse가 dense보다 개선되지 않음 | `outputs/reports/sparse_deep_residual_challenger_20260823/experiment_summary.json` |
| H5 rolling/fixed 8~15 연승 | 재사용 TEST에서만 제한 지지 | 일부 paired Holm 통과, 절대 ROI CI는 0 포함 | `outputs/reports/rolling_origin_market_challenger_20260823/experiment_summary.json`; `outputs/reports/fixed_lock_rolling_stability_20260823/experiment_summary.json` |
| H6 연승 calibration | 미지지 | proper score 일부 개선, Holm 0/4, calibrated Kelly 악화 | `outputs/reports/rolling_place_probability_calibration_20260823/experiment_summary.json` |
| H7 `clinic_30d` 증분 | 실제 ablation 후 미지지 | 40 clogit+32 ablation fits; 증분 CI 0 포함 | `outputs/reports/clinic30d_rolling_ablation_20260823/experiment_summary.json` |
| H8A initial27 | 불확정 | ΔLL +0.004139, day/race CI 0 포함, exact p=0.5 | `outputs/reports/fresh_minimal_entry_market_challenger_20260823/experiment_summary.json` |
| H8B additional37/pooled64 | 유망하지만 pristine 아님 | +0.028644/+0.018306, first27 확인 뒤 optional extension이고 4·6일뿐 | `outputs/reports/incremental_fresh_holdout_20260823/experiment_summary.json`; `outputs/reports/fresh_minimal_entry_market_challenger_extended_20260823/experiment_summary.json` |
| H9 fresh64 7승식 | 미지지 | 절대수익 0/7, paired Holm 0/7 | `outputs/reports/fresh_all_bets_extended_20260823/fresh_all_bet_summary.json`; `outputs/reports/fresh_all_bets_paired_advantage_extended_20260823/experiment_summary.json` |
| H10A 6피처 provenance | 기술 성공, 성능 미확정 | replay·감사 PASS, fresh64 exact p=0.078125 | `outputs/reports/h10a_provenance_six_feature_anchor_20260823/experiment_summary.json` |
| H11B 6피처 멀티모델 | 기술 동결 성공, 타깃 성능 미측정 | 실제 fit·reload·FD·확률합 PASS, 역사 CI 0 포함 | `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/experiment_summary.json` |
| H11C 8~15배 top10~40 | VALID 선택기간에서 미지지 | 네 cutoff 모델 ROI<0, q control>0, Holm 0 | `outputs/reports/h11c_darkhorse_8_15_preregistered_20260823/experiment_summary.json` |
| H11D alpha shrinkage | 미지지 | alpha 1 대비 증분 +0.000010245, CI·p 미확정 | `outputs/reports/h11d_market_anchor_shrinkage_preregistered_20260823/late_valid_metrics.json` |
| H13 nonnegative pool | 미지지 | q CI 0 포함, equal보다 점추정 열위 | `outputs/reports/h13_market_anchor_nonnegative_log_pool_preregistered_20260823/late28_metrics.json` |
| H14 convex pool | 제약 성공, 성능 미지지 | 제약·optimizer·확률합 PASS, 비교 CI 모두 0 포함 | `outputs/reports/h14_convex_market_residual_pool_preregistered_20260823/late28_metrics.json` |
| H17 ridge+one-SE | 안정화 성공, 증분가치 미지지 | 거의 q로 수축, q 대비 CI 0 포함 | `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/late28_metrics.json` |
| H11~H18 전향 governance | 지원됨, 성능 미측정 | 159파일·972/972·target/network/stake 0 | `outputs/reports/20260823_prospective_master_registry/independent_validation.json` |

## 4. 실패·사고·교정 이력

| 사건 | 실패 사실 | 교정과 채택 범위 | 근거 |
|---|---|---|---|
| Deep float32 softmax | 경주 확률합 오차 약 1.22e-7로 1e-10 gate FAIL | float64 경주 재정규화 후 v2 재학습·재검증; 첫 실행 비채택 | `outputs/reports/deep_market_challengers_20260823/independent_audit/independent_audit_summary.json`; `..._v2/independent_audit/independent_audit_summary.json` |
| 최종 pool 누수 | 과거 137피처에 `winAmt`, `plcAmt`, `totalAmt`, `liq_per_horse` 존재 | v11에서 최종 pool·배당·결과열 제거, 과거 결과는 배치근거 제외 | `Report_summary.txt`; `data/revised_v11_seoul_bugyeong_rank_clean_preprocessed/preprocessing_manifest.json` |
| 축퇴 배당 고ROI | +57.43%가 2019~2021 축퇴 구간과 일치 | 수익근거 무효 처리, 후속 원천·분포 감사 | `Report_summary.txt` |
| H11B determinism | 첫 validator 27/28; Deep diff VALID 5.891e-9, TEST 6.806e-9 > 1e-12 | tolerance 완화 없이 CPU thread=1·batch4096로 실행경로 일치; 최종 diff 3.331e-16, 28/28 | `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/independent_validation_incident_20260823.json` |
| H11D physical byte I/O | 전체 CSV hash 중 TEST bytes가 물리적으로 한 번 stream | physical byte와 decode/parse/materialize/metric/selection을 구분; 후자는 전부 0 | `outputs/reports/h11d_market_anchor_shrinkage_preregistered_20260823/physical_byte_io_disclosure.json` |
| H13 validator 비멱등 | read-only attestation overwrite로 `PermissionError` | `REUSE_NO_OVERWRITE` addendum; 학습·모델 불변 | `outputs/reports/h13_market_anchor_nonnegative_log_pool_preregistered_20260823/validator_idempotence_defect_before_fix.json`; `.../validator_idempotence_fix_addendum.json` |
| H14 AST 오탐 | `dict.get`을 HTTP GET으로 보아 71/72 FAIL | 네트워크 AST 탐지 구체화; 실패 보존; 최종 72/72 | `outputs/reports/h14_convex_market_residual_pool_preregistered_20260823/independent_validation_attempt1_failure.json` |
| H14 ephemeral validation hash | `validated_at_kst` 때문에 rerun SHA 변경 | stable normalized v3·append-only remediation, legacy dependency 영구 제외 | `outputs/reports/h14_prospective_t5_scoring_20260823/pre_t5_binding_remediation_addendum.json` |
| H11 check count 의미 혼동 | quote capture 170을 scorer 검증수로 오인할 위험 | scorer는 `216+2A` 동적 계약으로 교정 | `outputs/reports/authorized_postrace_pipeline_preregistered_20260823/pre_t5_h11_dynamic_final_gate_correction_addendum.json` |
| Combined cross-audit v1 false negative | same-line AST 순서를 line number만 비교해 207/208 | `(lineno,col_offset)` 비교 v2 208/208; frozen component bytes 불변 | `outputs/reports/h16_h18_settlement_cross_component_audit_20260823/cross_component_audit.json`; `.../false_negative_correction_addendum_v2.json`; `.../cross_component_audit_v2.json` |
| Orchestrator 조기 binding hold | authoritative cross-audit 전 component binding 생성 | 기존 binding 보존, fail-closed hold, v2 PASS exact release; production 미실행 | `outputs/reports/authorized_postrace_pipeline_preregistered_20260823/combined_binding_pending_cross_audit_hold_before_first_t5.json`; `.../combined_binding_cross_audit_release_before_first_t5.json` |

## 5. 저장된 문헌과 현업 방법론

이번 작업은 외부 원문을 새로 조회하지 않았다. 아래는 기존 구조화 목록과 보고서에 저장된 문헌 해석을 재사용한 것이므로, 로컬 카탈로그 존재는 `HIGH`, 외부 사실의 현재성·원문 정확성은 `MEDIUM`이다.

정본 카탈로그:

- `outputs/reports/interim_market_error_research_20260822/external_sources.json` — 21개 항목, SHA-256 `1F0691609DEBEAC5BBA85647BF10500670C2FBC7E3B045980CC88F5A14DF326A`
- `outputs/reports/market_probability_advantage_deep_research_20260823/index.html` — 문헌·현업·유사 프로젝트 절, SHA-256 `B6B0B1F8FCCB1D7C68D2CAA8D69D8A13CBB05BAF64403B74D2319B81EA5287F7`

| 문헌군 | 저장된 방법 | 이 프로젝트에 준 영향 | 한계 |
|---|---|---|---|
| William Benter | fundamental model과 public market probability OOS 결합, calibration, advantage, Kelly | `log(q)` offset/Base Margin, ΔLL, EDGE, fractional Kelly, 복합승식 보정 | 확률 외 실행가격·pool impact·자금배분 필요 |
| Ziemba | pari-mutuel 효율, 신디케이트, 리베이트, 확률·포트폴리오 최적화 | 비용·위험·capacity를 예측과 분리 | 리베이트 시장 경제를 KRA 무리베이트 성과로 전용 불가 |
| Uhrín; Despons·Peliti·Lacoste | adaptive/fractional Kelly, Bayesian adaptation, 초기 regret | 작은 Kelly, 노출상한, 불확실성 반영 | 검증 안 된 확률에는 Kelly가 손실 확대 |
| 한국 KRA LTR 연구(2024·2025) | RankNet, XGB, LGBM, CatBoost, pairwise/listwise, SHAP | race QID 그룹과 NDCG·Top3·rank MAE | 순위 정확도는 calibration·ROI가 아님 |
| Sugiura 2026 | leakage-aware race-level upset-risk layer | race-level abstention/risk gate 후보 | ROC-AUC·이변률은 ROI 증거가 아님 |
| Hanyu et al. | 막판 odds trajectory와 favorite-longshot bias | T-5 의사결정 snapshot 필요 | 기존 보고서상 preprint, 동료심사 완료로 간주하지 않음 |
| Gonçalves; Terawong·Cliff | 거래소 호가 attention/RNN, dynamic order placement | 가격경로·latency·시장충격 후보 | Betfair/합성 거래소와 KRA pool 구조 상이 |
| Harville 1973 | 단승확률 조건부 재정규화로 순서조합확률 | 7승식 probability-only 기준선 | IIA·2/3착 편향, 실제 조합가격 부족 |
| Gneiting·Raftery | strictly proper scoring rules | q 대비 ΔLL을 1차 확률지표로 사용 | proper score 개선이 ROI를 자동 보장하지 않음 |
| Snowberg·Wolfers; 한국 효율; Meyer·Hundtofte | favorite-longshot bias와 배당 맥락 | 8~15배를 보존하되 보편법칙으로 두지 않음 | 시장·시대·정의 차이로 직접 계수 이식 불가 |
| Market-model profitability + OSS | 시장가격 고려 model selection과 공개 재현코드 | VALID ΔLL와 paired q control | 외부 수익을 KRA에 직접 전용 불가 |
| White; Hansen; Romano·Wolf | Reality Check, SPA, stepwise multiple testing | BH FDR·Holm FWER, 전략 ledger 필요 | H13~H17 적응형 연쇄 전체 family 교정은 미실시 |

현업 회사·시장 구조 역시 “돈 버는 원리”와 “우리 모델의 입증 수익”을 구분해야 한다.

| 주체 | 기존 보고서에 저장된 사업 방식 | 프로젝트 함의 | 주의 |
|---|---|---|---|
| 전문 CAW 팀 | 작은 공정확률-시장가격 차이 + rebate + 대량회전 + 저지연 tote | 무리베이트 ROI, 속도, pool 대비 stake, 막판 odds drift 측정 필요 | 공개 예시는 독립감사 장기 ROI가 아님 |
| CAW/ADW 플랫폼 | handle 기반 서비스·host fee와 인프라 | 플랫폼비·접근권·주문제약 분리 | 플랫폼 매출 ≠ 고객팀 ROI |
| 경마장·시행체 | takeout, host fee, simulcast, media, commingling | 시장별 공제율·유동성·capacity·환급식 고정 | HKJC/World Pool 수치를 KRA에 대입 불가 |
| Betfair/Flutter | 이용자 간 back/lay, 순이익 commission | 체결률·CLV·호가깊이와 pari-mutuel 차이 분리 | exchange 방법을 KRA에 그대로 적용 불가 |
| 데이터·토트 기술사 | 데이터·가격·tote·리스크관리 B2B | API SLA·QA·latency가 운영지표 | B2B 매출 ≠ 알고리즘 ROI |
| 조합 pool·cash-out | pool 공제, jackpot, cash-out pricing | 생존확률·공정 cash-out·hedge 비용 | 회사 공식 설명은 독립 성과감사 아님 |

법률 경계는 기존 보고서가 저장한 `한국마사회법 제48조` 경고를 따른다. 이 문서는 법률 자문이 아니고 자동주문 실행 가능성은 별도 전문 검토가 필요하다.

## 6. 핵심 용어와 수식

| 용어 | 정확한 의미·수식 |
|---|---|
| 시장확률 `q` | 한 경주 표시 단승 gross odds `o_i`로 `q_i=(1/o_i)/Σ_j(1/o_j)`. 경주합 1. |
| 시장 앵커/Base Margin | 모델 score를 `log(q_i)+f(x_i)`로 두는 방식. `log(q)`는 ordinary feature가 아니라 고정 offset이며 `x`에서 제외. |
| Residual | 시장 log-odds가 놓친 비시장 정보 `f(x)`. |
| Softmax | `p_i=exp(s_i)/Σ_j exp(s_j)`. 반드시 경주 안에서 합 1. |
| Log loss/NLL | 우승마 `y`에 대해 `-log p_y`; 낮을수록 좋음. |
| ΔLL | `LL_market-LL_model` 또는 동등한 부호계약. 이 프로젝트 보고서는 양수를 모델 우위로 고정. |
| Brier | `Σ_i(p_i-y_i)^2`; 확률 오차의 제곱합, 낮을수록 좋음. |
| ECE | 예측확률 bin별 평균확률과 실제빈도의 가중 절대차. calibration 진단. |
| EDGE | 기본적으로 `p_model-q_market`; 가격비 표현은 `p_model*odds-1`; 어느 정의인지 보고서별 명시 필요. |
| EV | 1단위 stake에서 `EV=p*net_payout-(1-p)*loss`; 배당 정의·수수료 이중차감 금지. |
| ROI | `(총환급-총stake)/총stake`. 절대 ROI와 동일건수 q 대조의 paired ΔROI를 분리. |
| Hit Rate | 적중 티켓 수/전체 티켓 수. ROI와 동일하지 않음. |
| MDD | 자본곡선의 이전 고점 대비 최대 낙폭. |
| Sharpe | 평균 초과수익/수익 변동성; 이 프로젝트에서는 일자 block 정의 확인 필요. |
| Kelly | 공정확률 `p`, decimal odds `b+1`일 때 `f*=(bp-(1-p))/b`; 추정오차 때문에 fractional Kelly 사용. |
| Validation | 모델·임계값·정책 선택 구간. TEST와 섞으면 안 됨. |
| TEST | 잠금 정책의 OOS 평가 구간. 이 프로젝트 TEST는 반복 열람돼 pristine confirmation이 아님. |
| Untouched forward OOS | 가설·코드·임계값·파일 hash가 결과 전에 고정된 새 미래 표본. |
| Block bootstrap CI | 날짜/경주 묶음을 재표집해 시계열·군집 의존성을 보존한 신뢰구간. |
| Paired sign-flip | 같은 날짜의 모델-시장 차이 부호를 무작위 반전하는 대응 검정. |
| Holm FWER | 여러 가설 중 하나 이상 거짓양성 확률을 제어하는 step-down 보정. |
| BH FDR | 발견 집합 내 기대 거짓발견비율을 제어. |
| NDCG@5 | 실제 상위 착순에 높은 relevance를 주고 예측 상위 5의 discounted gain을 이상순위로 정규화. |
| Spearman | 실제·예측 순위의 단조 상관. 1에 가까울수록 순서 일치. |
| Rank MAE | 말별 `|예측착순-실제착순|` 평균. |
| Ordered/Unordered Top3 | ordered는 1·2·3착과 순서까지 일치, unordered는 세 말 집합만 일치. |
| Gradient/Hessian | 손실의 1차/2차 미분. custom tree objective와 analytic optimizer 검증에 사용. |
| SLSQP | 경계·선형 제약을 포함해 연속 가중치를 최적화한 수치 알고리즘. |
| Residual log-pool | `p∝q*exp(Σ_k w_k log(p_k/q))`. |
| Convex pool | 위 식에서 `w_k>=0`, `Σw_k<=1`; 시장 지수 `1-Σw_k`가 음수가 되지 않음. |
| Ridge/one-SE | `λ||w||²`로 수축하고 최저 DEV 손실에서 1SE 이내인 더 단순 λ를 선택. |
| Plackett-Luce/Harville | 남은 말 확률을 조건부 재정규화해 순서조합확률을 계산. |
| Pari-mutuel/takeout | 총 pool에서 공제 후 적중자에게 분배. 공식 payout이면 수수료 재차감 금지. |
| Slippage | 의사결정 가격과 실제 체결·최종 가격의 불리한 차이. |
| CAW/ADW | computer-assisted wagering / advance-deposit wagering 인프라. |
| OSS | 공개소스 소프트웨어. 코드 공개는 재현성을 돕지만 외부 성과의 우리 시장 재현을 보장하지 않음. |
| Preregistration/freeze | 결과 전 모델·대상·임계값·지표·hash를 고정하고 byte 변경을 거부. |
| Fail-closed | 권한·시각·hash·완결성·계약 중 하나라도 불확실하면 실행하지 않음. |

## 7. 핵심 생성 파일과 SHA-256

아래는 최종 보고서 생성기가 우선 읽을 정본이다. 34개 전체 목록은 JSON의 `key_generated_files`에 있다.

| 역할 | 경로 | SHA-256 |
|---|---|---|
| 상시 운영지침 | `../AGENTS.md` | `70DBF4D065387C1EFAA6C4FF60610F7226D93A38577EBF767CFF4C2122E31155` |
| 파일·증거 인벤토리 | `Project_Reserach.txt` | `0F8A2E5ED5524215F68543D44972297724236636893BFBC382E132F334C897BF` |
| 기준 보고서 요약 | `Report_summary.txt` | `040B6CE1FB146573E43F67E4A8814C0A428333C50DA22C823C52E3DF6A32E1A7` |
| v11 전처리 manifest | `data/revised_v11_seoul_bugyeong_rank_clean_preprocessed/preprocessing_manifest.json` | `309AE27C17535364B21443FBDA2C8333AEDDC9A6ED3D375DCD27A20063D93F25` |
| v11 데이터 검증 | `data/revised_v11_seoul_bugyeong_rank_clean_preprocessed/preprocessing_validation.json` | `C3B36201429B7067C6F7C84E3E43610DA0267519605815E066E50AB8F227DE98` |
| v11 203개 최종검증 | `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/final_validation.json` | `B9CB967F333565AB3BA331FAE0C7FBD08825857243BA5ED34D5826A47F1E21A2` |
| H1~H9 snapshot | `outputs/reports/market_probability_advantage_deep_research_20260823/report_data_snapshot.json` | `8C8C56FE2B4F8B10B43BBBD917A181C1D4EC1AEAA6DCB3CAF0763434E29479B3` |
| H1~H9 HTML | `outputs/reports/market_probability_advantage_deep_research_20260823/index.html` | `B6B0B1F8FCCB1D7C68D2CAA8D69D8A13CBB05BAF64403B74D2319B81EA5287F7` |
| H1~H9 PDF | `outputs/reports/market_probability_advantage_deep_research_20260823/index.pdf` | `F9D7246BBDA679DE6ED3556C5AE1E50FE09F3C2CBFD0A7E435BB4990A2F0BFC0` |
| 외부자료 21개 catalog | `outputs/reports/interim_market_error_research_20260822/external_sources.json` | `1F0691609DEBEAC5BBA85647BF10500670C2FBC7E3B045980CC88F5A14DF326A` |
| H10A summary | `outputs/reports/h10a_provenance_six_feature_anchor_20260823/experiment_summary.json` | `FAD2AD5247E19964819FEAC07C9244E8ED69A7B459F96929C8CB96DC760AACA3` |
| H11 V1 prereg | `data/raw_prospective_oos_20260823/preregistration.json` | `26EB3D7B691A06647994166DC20702ECCD641803FFC95B4AB02764A44F2FA374` |
| H11B protocol | `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/protocol_before_fit.json` | `39B235D33C203F00406DF612EAA3FF146B6D1850101EA7D5AC71F9C75EEE551B` |
| H11B summary | `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/experiment_summary.json` | `25B5D46725763BA394C1EEADEB7F9A20DBDFEBCAA1669250AF5FAC064FD9C3C4` |
| H11B 28/28 | `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/independent_validation.json` | `021A9119CEB39E2BBE73F5F0FA50DB48D2A8D022353DE5CA5A7B54F91C78D084` |
| H11C threshold lock | `outputs/reports/h11c_darkhorse_8_15_preregistered_20260823/threshold_lock.json` | `C8952FE023BA53D16453171AD1AF5B5D9AFA037D00C741DF2A30F24BB0BABC63` |
| H11D protocol | `outputs/reports/h11d_market_anchor_shrinkage_preregistered_20260823/protocol_before_analysis.json` | `E6D5C6D960BA391C6E00926C43090DFC595BB63886B3B83160E394D96737D873` |
| H11D alpha freeze | `outputs/reports/h11d_market_anchor_shrinkage_preregistered_20260823/frozen_alpha.json` | `B2BF4F77F14A06ACAD5502BC9AF9B90287C80502ECCB773872DBC3D961A0579C` |
| H13 summary | `outputs/reports/h13_market_anchor_nonnegative_log_pool_preregistered_20260823/experiment_summary.json` | `24D54C8B82DF2580C96FA783410C552C478545F82329B5B443A73E000A58CE05` |
| H14 summary | `outputs/reports/h14_convex_market_residual_pool_preregistered_20260823/experiment_summary.json` | `E2B2105022C7EDF9BB3B13149CB9831577F74A246AEFE4A1DAEC9C12C159AFFD` |
| H14 stable 72/72 | `outputs/reports/h14_convex_market_residual_pool_preregistered_20260823/independent_validation_frozen_before_first_t5_v3.json` | `7C5625003D0EBA093FDECB250C5FF8BAAFB0C7021101F31ED0FD7C16267818C4` |
| H17 protocol | `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/protocol_before_training.json` | `5E19A19CA41837CAFDB9ABEADC0AA5958EEAC14C8CA17F0323592F530F747A78` |
| H17 summary | `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/experiment_summary.json` | `C35CC9D66CA42778D6C08432905D03C62BBA9239406CAA13ED006A9E784B64BA` |
| H17 stable 268/268 | `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/independent_validation_frozen_before_first_t5.json` | `C0DD7EF62400B451CE05314733CBABD1CAE654CD5FB62BC6EFE21C5895D6BB0A` |
| H11~H18 master registry | `outputs/reports/20260823_prospective_master_registry/master_registry.json` | `7F483003B1DFD66F7784C4D3AF498DA7A3A37AEA09ADFDC550D88CF1FA5EB5E2` |
| Registry 972/972 | `outputs/reports/20260823_prospective_master_registry/independent_validation.json` | `1BCD2E4A8F48ADC3CDE228D004C7E034D0E69C97D844438829075720C8634CAE` |
| Combined cross-audit | `outputs/reports/h16_h18_settlement_cross_component_audit_20260823/cross_component_audit_v2.json` | `2802DF250FA7296F863BFE37BBE091DE02EF0ABA45D6C1FC91EE900189D860AA` |
| Orchestrator protocol | `outputs/reports/authorized_postrace_pipeline_preregistered_20260823/protocol_before_first_t5.json` | `DB0214BD57BA4BEFBC119AB7D10B21E2AF46803243D713851AE155A9283BABAB` |
| Orchestrator final freeze | `outputs/reports/authorized_postrace_pipeline_preregistered_20260823/freeze_complete_cross_audited_before_first_t5.json` | `245FB43AC8D16213C11B284C4EA37FAB97FDFC7A50E25E3F0093EA32B04699D7` |
| Orchestrator 71/71 | `outputs/reports/authorized_postrace_pipeline_preregistered_20260823/independent_validation_final_cross_audit_v5_frozen.json` | `2BF415F9E385C66BDE515C200D557BD598E526244FE1E605E8776B0FBCD2A540` |

## 8. 재사용 가능한 시각자료

새 차트를 만들 필요 없이 다음 기존 자산을 재사용할 수 있다. 모든 경우 원래 scope warning을 함께 표시해야 한다.

- `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/complete_report/index.html`
  - inline SVG: 6개 모델 TEST NDCG@5
  - inline SVG: 시장 앵커 TEST ΔLL/race와 날짜 block 95% CI
  - inline SVG 2개: 전체착순·시장앵커 7승식 TEST ROI
  - 경고: TEST 반복 열람, 복합승식 proxy/가격 한계.
- `outputs/reports/market_probability_advantage_deep_research_20260823/index.html`
  - inline SVG: 시간순 OOS ΔLL, paired ΔROI, partial30 중요도, Deep MLP permutation 중요도
  - 경고: fresh/재사용/optional continuation 라벨 유지, 중요도는 인과 아님.
- `outputs/reports/h11b_six_feature_multimodel_freeze_report_20260823/index.html`
  - inline SVG: M6/XGB/LGBM/Deep/ensemble VALID·TEST ΔLL와 CI
  - 경고: 역사 진단이지 목표일 성능 아님.
- `outputs/reports/h11c_darkhorse_8_15_valid_lock_report_20260823/index.html`
  - inline SVG 2개: top10~40 모델 대 q ROI와 paired CI
  - 경고: VALID 선택기간 반증 결과.
- `outputs/reports/h11d_market_anchor_shrinkage_report_20260823/index.html`
  - inline SVG: alpha별 DEV·all VALID ΔLL 곡선
  - 경고: late VALID CI·p는 우위 미확인.
- `outputs/reports/market_probability_advantage_deep_research_20260823/index.pdf` — 42쪽 visual QA PASS.
- `outputs/reports/project_learning_textbook_20260823/index.pdf` — 21쪽 개념 교재. H10~H18은 미포함.
- `outputs/reports/revised_v7_full_rank_rerun_20260820/full_rank_model_comparison.png` — 과거 서울-only v7 부록 전용.

## 9. 보고서·산출물 검증 이력

| 산출물 | 검증 결과 | 근거 |
|---|---|---|
| v11 모델 | 203/203 PASS | `outputs/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/final_validation.json` |
| v11 종합 | 20/20 PASS | `.../comprehensive_validation.json` |
| v11 HTML | 77/77 PASS, 표 31·SVG 4·근거링크 67 | `.../complete_report/html_validation.json` |
| v11 VS 노출 | 26/26 PASS; browser visual QA는 보안정책 차단 | `.../complete_report/visual_studio_visibility.json` |
| 중간 문헌 HTML | 15/15 PASS | `outputs/reports/interim_market_error_research_20260822/html_validation.json` |
| 개념교재 PDF | 21페이지 render·visual QA PASS | `outputs/reports/project_learning_textbook_20260823/pdf_validation.json` |
| H1~H9 HTML/PDF | HTML 12/12; PDF 42쪽 전부 PASS | `outputs/reports/market_probability_advantage_deep_research_20260823/independent_html_validation.json`; `.../pdf_validation.json` |
| H10A PDF | 19/19, 7쪽 PASS; malformed 15쪽은 거부 후 교정 | `outputs/reports/h10a_provenance_six_feature_final_report_20260823/pdf_validation.json` |
| H11B core/PDF | core 28/28; PDF 28/28, 6쪽 PASS | `outputs/reports/h11b_six_feature_multimodel_preregistered_20260823/independent_validation.json`; `outputs/reports/h11b_six_feature_multimodel_freeze_report_20260823/pdf_validation.json` |
| H11C | analysis/report 62/62; PDF 34/34, 5쪽 PASS | `outputs/reports/h11c_darkhorse_8_15_valid_lock_report_20260823/validation.json` |
| H11D | core 37/37; HTML 21/21; PDF 23/23, 5쪽 | `outputs/reports/h11d_market_anchor_shrinkage_preregistered_20260823/independent_validation.json` |
| H13 | 40/40 PASS, idempotence 결함 보존·교정 | `outputs/reports/h13_market_anchor_nonnegative_log_pool_preregistered_20260823/independent_validation_complete.json` |
| H14 | stable 72/72 PASS_WITH_LIMITATIONS | `outputs/reports/h14_convex_market_residual_pool_preregistered_20260823/independent_validation_frozen_before_first_t5_v3.json` |
| H17 | stable 268/268 PASS | `outputs/reports/h17_ridge_convex_market_residual_pool_preregistered_20260823/independent_validation_frozen_before_first_t5.json` |
| Master registry | 972/972 PASS, 159 files | `outputs/reports/20260823_prospective_master_registry/independent_validation.json` |
| Combined cross-audit | 208/208 + metrics 2820/2820 + negative 35/35 | `outputs/reports/h16_h18_settlement_cross_component_audit_20260823/cross_component_audit_v2.json` |
| Authorized orchestrator | 71/71 두 번, 두 번째 `REUSE_NO_OVERWRITE`; production 미실행 | `outputs/reports/authorized_postrace_pipeline_preregistered_20260823/independent_validation_final_cross_audit_v5_frozen.json` |

## 10. 반드시 `미측정`으로 표시할 지표

다음 값을 최종 보고서에서 0이나 실패로 채우면 안 된다. 측정하지 않았거나 식별 불가능하다.

- 2026-08-23 H11/H11B/H13/H14/H17 winner ΔLL·NLL·Brier·ECE·Top1: `NOT_MEASURED` — 결과 미조회, evaluator 미실행.
- 2026-08-23 ordered/unordered Top3·position accuracy: `NOT_MEASURED`; full order가 없으면 NDCG·Spearman·rank MAE는 `N/A`.
- 2026-08-23 7승식 실제 ROI·Hit Rate·평균배당·MDD·Sharpe: `NOT_MEASURED` — production 정산 미실행, stake 0.
- 2026-08-23 H11C 8~15배 top10/20/30/40 전향수익: `NOT_MEASURED` — 임계값만 잠김.
- H11C stopping rule 20일·500후보: `NOT_REACHED`.
- 모든 복합승식 T-5 q·EDGE: `PARTIALLY_UNAVAILABLE` — exact complete quinella grid 외 전체 가격 grid 없음.
- 복합승식 actual EV·Kelly: `NOT_IDENTIFIABLE` — 의사결정 시점 전체 조합 odds/pool 부족.
- 실측 slippage·latency·odds drift: `NOT_MEASURED`; 기존 2~20% haircut은 시나리오.
- 자기 주문의 pool impact·capacity: `NOT_MEASURED`.
- 리베이트 포함/미포함 순수익 분해: `NOT_MEASURED`.
- 여러 pristine 미래날짜의 장기 ROI·안정성: `NOT_MEASURED`.
- H13~H17 적응형 연쇄 전체 family의 통합 다중검정: `NOT_MEASURED`.
- 피처의 인과효과: `NOT_MEASURED`; SHAP·계수·permutation은 연관 설명.
- Monte Carlo 조합확률의 실제 조합시장 calibration: `NOT_MEASURED`.
- 회사·전문 베팅팀의 독립 감사 장기 ROI: `NOT_AVAILABLE_IN_SAVED_SOURCES`.
- 법률·약관상 자동주문 가능성: `NOT_DETERMINED`.

## 11. 최종 보고서 작성 지침

최종 보고서는 다음 순서가 증거의 위계를 가장 잘 보존한다.

1. 판정 언어와 지표 정의
2. 데이터 무결성과 시간분할
3. 전체 착순 6모델
4. Benter·시장 앵커 방법론
5. H1~H9 반복가설과 반증
6. H10A~H17 단순화·pooling 실험
7. 7승식 절대수익과 시장대조
8. T-5 전향 사전등록 인프라
9. 사고·교정·독립검증 이력
10. 문헌·현업 방법론과 프로젝트 대응
11. 미측정 지표와 NO-GO 경계

한 문장 결론은 다음보다 강하게 쓰면 안 된다.

> 이 프로젝트는 재사용 역사표본에서 시장확률에 추가되는 작은 정보 신호와 일부 방어적 paired 우위를 찾았지만, clean하고 독립적인 장기 전향 표본에서 수수료·실행가격·시장충격 후 절대수익을 확인하지 못했으므로 현재는 연구 인프라 PASS, 실전 배치 NO-GO, stake 0이다.

