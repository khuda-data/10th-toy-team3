# Benter 방식 이변·8~15배 역사적 재현 패키지

이 폴더는 10시간 장기 연구 이전에 수행한 시장 앵커, 이변 말 탐색, 8~15배 구간 분석을 재현하기 위한 **누락 자료만** 모은 패키지다. 새 `index` 보고서를 만들지 않았고 기존 보고서는 원래 Git 경로를 참조한다.

## 당시 핵심 관찰

- Benter식 시장 앵커 조건부 로짓은 시장확률 `q`를 버리지 않고 `ln(q)`를 기준선으로 둔 뒤, 시장이 덜 반영한 피처의 잔차정보만 학습했다.
- 역사적 잠금 백테스트의 8~15배·EDGE 상위 5% 정책은 327건에서 ROI `+32.69%`였다. 같은 배당대에서 시장확률만으로 고른 대조군은 `-3.33%`였다.
- 그러나 ROI 95% CI는 `[-2.70%, +68.25%]`로 0을 포함했다. 따라서 흥미로운 시장 대비 신호였지만 통계적으로 확정된 수익 전략은 아니다.
- 후속 v11 재검증은 94건 ROI `+9.79%`, 동일 구간 시장 대조군 `-12.45%`였으나 CI가 넓어 역시 수익 확정은 아니었다.

## 기존 Git 보고서 — 중복 복사하지 않음

- `../final_report_and_models_20260823/reports/benter_market_anchored_20260820/돌파구_보고서.pdf`
- `../final_report_and_models_20260823/reports/benter_market_anchored_20260820/돌파구_보고서.md`
- `../final_report_and_models_20260823/reports/odds_8_15_v11_revalidation_20260823/index.pdf`
- `../final_report_and_models_20260823/reports/odds_8_15_v11_revalidation_20260823/index.html`
- `../final_report_and_models_20260823/reports/interim_market_error_research_20260822/index.pdf`

## 데이터셋 — 기존 파일 재사용

최초 Benter 워크포워드 입력 `final.csv.gz`는 현재 저장소의 `../data/race_entries.csv.gz`와 바이트 단위로 같다.

- 크기: `23,285,667` bytes
- SHA-256: `964BD9A7AB7E36247FC5A7E5FFD04F9EC02439491B1DACAEF7918EB0AAE80195`

따라서 같은 데이터를 다른 이름으로 다시 올리지 않았다.

## 이 폴더에 새로 추가한 모델·보고서

- `models/market_anchor_same_test_20260822`: 과거 동일-test 시장 앵커 Base Margin 및 조건부 로짓 저장 모델
- `models/upset_feature_experiment_20260822`: 전체 피처 및 선별 피처 이변 Base Margin 저장 모델
- `reports/upset_feature_experiment_20260822`: 이전 결과, 전체 피처, 선별 피처 재학습, 독립 검증 보고서와 예측·중요도·EDGE 자료
- `reports/benter_two_step_20260821`: Benter 2단계 후속 보고서
- `reports/base_margin_boosting_20260821`: Base Margin 후속 보고서
- `src`: 시장 앵커·백테스트·2단계·Base Margin·이변 실험 및 독립검증 코드

최초 2026-08-20 워크포워드는 fold별 적합 계수를 별도 바이너리 모델로 저장하지 않았다. 해당 실행의 모델 산출물은 기존 `benter_market_anchored_20260820` 폴더의 OOS 확률, fold 결과, 잠금 정책과 베팅 결과다. 바이너리 모델이 있었다고 소급해 표현하지 않는다.

## 중복 감사

- 추가 후보 50개를 현재 Git `HEAD`의 모든 blob과 비교했다.
- 내용이 완전히 같은 파일은 `reproduction_commands.txt` 1개뿐이어서 추가하지 않았다.
- 기존 동일 파일: `../final_report_and_models_20260823/reports/revised_v11_seoul_bugyeong_full_rerun_20260822/upset_feature/reproduction_commands.txt`
- 데이터셋도 동일 SHA-256이므로 추가하지 않았다.
- 나머지 보고서·모델·코드 49개만 이 폴더에 추가했다.
