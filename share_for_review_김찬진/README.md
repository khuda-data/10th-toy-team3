# 서울 경마 시장확률 보정 프로젝트

한국마사회 서울 경주 데이터를 이용해 개별 출전마의 승리확률을 추정하고, 그 확률이 최종 배당으로 계산한 시장 확률보다 더 정확한지 검증하는 AI 동아리 토이프로젝트다.

이 프로젝트의 목표는 단순 적중률을 높이는 것이 아니다.

> 시간순으로 분리한 미래 경주에서 모델이 시장보다 낮은 Race Log Loss와 Race Brier Score를 기록하며, 그 개선이 경주 단위 신뢰구간에서도 재현되는지 검증한다.

수익성은 별도 문제로 취급한다. 확률 예측이 조금 개선되어도 공제율을 극복하지 못하면 베팅하지 않는다.

## 현재 결론

사전에 고정한 최종 후보는 다음과 같다.

```text
model: M2 XGBoost
pre-race normalization: sum
market blend lambda: 0.05
temperature: 0.95
model version: m2_xgboost_sum_l005_t095_v1
```

Final Test는 2025-12-28부터 2026-08-09까지의 635경주를 최초 1회 평가했다.

| 후보 | Race Log Loss ↓ | Race Brier ↓ | Top-1 |
|---|---:|---:|---:|
| 시장 M0 | 1.817348 | 0.777920 | **37.80%** |
| 고정 M2 후보 | **1.813446** | **0.777318** | 37.48% |

- Log Loss 개선: `+0.003902`, bootstrap 95% CI `[+0.001085, +0.006721]`
- Brier 개선: `+0.000602`, bootstrap 95% CI `[-0.000483, +0.001687]`
- Log Loss 개선은 통계적으로 지지되지만 Brier 신뢰구간은 0을 포함한다.
- 엄격한 종합 성공 조건은 아직 충족하지 못했다.
- 최종 배당 기준 양의 기대수익 후보가 없어 운영 베팅 정책은 `no_bet`이다.

이 결과는 연구 결과이며 실제 베팅 또는 금융 조언이 아니다.

## 프로젝트 구조

```text
data/
  raw/                 동결한 원본 데이터
  interim/             서울 데이터와 시간순 split
  manifests/           schema, 정책, checksum, 평가 잠금
  predictions/         Calibration/Test 예측과 출력 fixture
  analysis/            bootstrap 반복표본과 백테스트 선택 기록
artifacts/models/      적합된 전처리기와 모델 artifact
src/
  data/                로더, schema 검증, 서울 interim, split
  features/            feature registry와 Train 전용 전처리
  models/              M0/M1/M2, 정규화, 혼합, 보정, Final Test
  evaluation/          경주 지표, bootstrap, 경제적 백테스트
  inference/           운영 예측 출력 계약
tests/                 데이터·누수·모델·평가·출력 계약 테스트
reports/experiments/   단계별 JSON 결과와 한국어 요약
```

기존 `전처리 데이터셋/v1~v4`는 참고용이며 학습 입력으로 사용할 수 없다. `v5~v8`은 경주 일부 행을 제거해 경주 구조를 깨뜨리므로 사용 금지다. 자세한 정책은 `data/manifests/dataset_policy.json`에 있다.

## 빠른 시작

Python 3.12 환경을 권장한다. 자세한 설치 방법은 [SETUP.md](SETUP.md)를 참고한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Windows에서는 다음 스크립트도 사용할 수 있다.

```powershell
.\scripts\run_tests.ps1
```

## 데이터 계약

Canonical 데이터:

- 원본: `data/raw/final.csv.gz`
- 서울 interim: `data/interim/seoul_entries.csv.gz`
- 경주 split: `data/interim/split_manifest.csv`
- 피처 승인 목록: `data/manifests/feature_registry.json`

고정 split:

| Fold | 기간 | 경주 | 행 | 용도 |
|---|---|---:|---:|---|
| Train | 2023-08-05 ~ 2025-05-11 | 1,891 | 19,617 | 모델·전처리 학습 |
| Calibration | 2025-05-17 ~ 2025-12-27 | 641 | 6,582 | lambda·temperature·임계값 선택 |
| Final Test | 2025-12-28 ~ 2026-08-09 | 635 | 6,639 | 최초 1회 최종 평가 |
| Excluded | 비정상 승자 구조 | 5 | 50 | 모델링 제외 |

모델에는 feature registry에서 `PRE_RACE`로 승인한 112개 열만 입력한다. 배당·결과·legacy·split·ID·control 열은 독립 모델의 표준 피처로 사용할 수 없다.

## 파이프라인

```text
원본 동결 및 검증
  → 서울 경주 추출
  → 경주 단위 시간순 split
  → 누수 방지 feature registry
  → Train 전용 결측·clipping·encoding
  → M0 시장 / M1 Logistic / M2 XGBoost
  → 경주 단위 sum 정규화
  → 시장 기하혼합 lambda=0.05
  → Temperature Scaling T=0.95
  → Final Test 최초 1회 평가
  → 경주 단위 paired bootstrap
  → closing-odds 경제적 백테스트
  → 예측 출력 schema
```

개발 원칙과 수식의 전체 내용은 [PROJECT_GUIDELINES.md](PROJECT_GUIDELINES.md)를 따른다.

## 재현 명령

아래 명령은 저장된 결과를 처음부터 재구축하는 개발용 순서다.

```powershell
python -m src.data.validate_schema
python -m src.data.build_seoul_interim
python -m src.data.build_splits
python -m src.features.build_registry
python -m src.models.market_baseline
python -m src.models.train_m1_logistic
python -m src.models.train_m2_xgboost
python -m src.models.select_normalization
python -m src.models.select_market_blend
python -m src.models.select_temperature
```

### Final Test 잠금

현재 저장소의 Final Test는 이미 평가됐다. 다음 명령은 일회성 잠금 때문에 다시 실행하면 실패하는 것이 정상이다.

```powershell
python -m src.models.freeze_final_test
python -m src.models.evaluate_final_test
```

Test 결과를 본 뒤 모델·피처·정규화·lambda·temperature를 변경하면 현재 Test는 더 이상 유효한 Test가 아니다. 새 실험은 새로운 미래 holdout을 확보해 별도 버전으로 시작해야 한다.

## 예측 출력 사용

`src.inference.prediction_contract.generate_predictions()`는 완전한 경주별 사전 피처와 배당 snapshot metadata를 입력받는다.

필수 metadata:

```text
winOdds_snapshot
odds_snapshot_time
prediction_time
race_start_time
odds_source
```

주요 검증:

- 선언 출전두수 `dusu`와 실제 입력 행 수 일치
- `odds_snapshot_time <= prediction_time < race_start_time`
- 경주별 `q_market`, `p_premarket`, `p_final` 합이 1
- `entry_id` 유일
- 불완전한 경주는 경주 전체 `prediction_rejected`
- 현재 유효 예측의 행동은 베팅 정책에 따라 모두 `no_bet`

출력 schema는 `data/manifests/prediction_output_schema.json`, 예시는 `data/predictions/stage_18_contract_fixture.csv`에서 확인할 수 있다. 예시 파일은 역사적 최종 배당과 합성 timestamp를 사용하므로 라이브 예측이 아니다.

## 주요 결과 문서

- `reports/experiments/stages_8_11_summary.md`: 전처리와 M0/M1/M2
- `reports/experiments/stage_12_summary.md`: 경주 확률 정규화
- `reports/experiments/stage_13_summary.md`: 시장 기하혼합
- `reports/experiments/stage_14_summary.md`: Temperature Scaling
- `reports/experiments/stage_15_summary.md`: Final Test
- `reports/experiments/stage_16_summary.md`: bootstrap 신뢰구간
- `reports/experiments/stage_17_summary.md`: 경제적 백테스트
- `reports/experiments/stage_18_summary.md`: 예측 출력 계약
- `reports/experiments/stage_19_summary.md`: README와 테스트 체계
- `reports/experiments/stage_20_summary.md`: 최종 결과보고서
- `reports/experiments/stage_21_summary.md`: Top-1 제약 연구 정책 동결
- `reports/experiments/stage_22_summary.md`: Train OOF·Calibration 시장/M2 1순위 불일치 분석
- `reports/experiments/stage_23_summary.md`: XGBoost 랭커용 경주 그룹 데이터 계약
- `reports/experiments/stage_24_summary.md`: R2 pairwise ranker 시간순 학습과 순위 성능 비교
- `reports/experiments/stage_25_summary.md`: R2 점수의 경주 softmax 확률 변환과 temperature 선택
- `reports/experiments/stage_26_summary.md`: 시장 유지/교체 게이트와 경주별 lambda 선택
- `reports/experiments/stage_27_summary.md`: 비열화 제약 후보 비교와 Future Holdout 도전자 동결
- `FUTURE_HOLDOUT_VALIDATION.md`: 새로운 최종 성능 및 시장 우위 공식 검증 사전등록 절차

## Top-1 제약 연구 확장

21단계부터 시장보다 높은 Top-1 accuracy를 탐색하되, Calibration에서 시장 대비 Race Log Loss와 Race Brier가 악화되지 않는 후보만 비교한다. 기존 Final Test는 이미 공개됐으므로 새 후보 선택에는 사용하지 않는다. 공식 평가는 2026-08-09 이후 첫 500개 적격 서울 경주로 구성할 새 Future Holdout에서 후보 동결 후 한 번만 수행한다.

세부 데이터 경계, 동률 처리, 후보 선택 순서와 통계적 성공 기준은 `data/manifests/top1_research_policy.json` 및 `PROJECT_GUIDELINES.md` 29절을 따른다. 새 데이터가 충분하지 않은 동안 상태는 `pending_data`이며 기존 챔피언과 `no_bet` 정책은 유지된다.

22단계 분석에서는 Train OOF 770경주와 Calibration 641경주만 사용했다. 총 1,411경주 중 시장과 M2의 1위가 710경주에서 달랐으며 시장만 적중한 경우 242건, M2만 적중한 경우 116건이었다. 따라서 기존 M2의 1위를 무조건 채택하는 방식은 시장보다 126경주 불리했고, 이후 랭킹 모델과 보수적 게이트가 필요하다.

23단계에서는 Train 19,617행·1,891경주와 Calibration 6,582행·641경주를 `race_id`별 연속 그룹으로 변환했다. 독립 랭커에는 112개 `PRE_RACE` 피처만 제공하며 우승마 relevance는 1, 나머지는 0이다. 행 순서와 XGBoost `group` 배열의 대응은 `data/interim/ranking_entry_manifest.csv.gz`와 `data/interim/ranking_group_manifest.csv`로 검증한다.

24단계 R2 pairwise ranker는 Train OOF에서 Top-1 28.05%, Calibration에서 30.11%를 기록했다. 기존 M2보다 각각 1경주와 3경주 많았지만 시장보다 68경주와 54경주 적었고, M2 대비 개선도 네 fold에서 일관되지 않았다. R2 점수는 아직 확률이 아니므로 Log Loss·Brier 평가는 25단계 경주 softmax와 temperature 선택 이후에만 수행한다.

25단계에서 Calibration Race Log Loss로 `T_rank=0.65`를 선택했다. R2 확률은 기존 독립 M2보다 Calibration Log Loss `0.009432`, Brier `0.001890` 개선됐지만 시장보다 각각 `0.191283`, `0.057196` 나빴다. R2 단독은 비열화 제약을 통과하지 못하므로 시장 대체 후보가 아니며, 이후 시장 유지형 게이트의 보조 신호로만 사용한다.

26단계 게이트는 Train OOF 불일치 경주로 학습하고 Calibration에서 threshold `0.65`, `lambda_switch=0.30`을 선택했다. 33경주에서 R2 영향을 허용했지만 실제 시장 1위 교체는 14경주였으며, 시장 247/641에서 후보 249/641로 2경주 증가했다. Log Loss 개선 `+0.001795`, Brier 개선 `+0.000338`로 점추정 비열화 제약도 통과했다. 이는 Calibration 선택 결과일 뿐 공식 우위가 아니며 새 Future Holdout 전까지 예비 후보로만 유지한다.

27단계에서는 R0 시장, R1 기존 M2, R2 랭커, R3 고정 시장 혼합, R4 적응형 게이트를 동일한 641개 Calibration 경주와 고유 Top-1 규칙으로 비교했다. R0·R3·R4가 확률 비열화 제약을 통과했고, R4가 시장보다 Top-1 `+2`경주로 가장 높아 `r4_gate_ranker_t065_gate065_l030_v1` 도전자로 동결됐다. 상태는 `frozen_pending_future_holdout`이며 기존 챔피언과 `no_bet` 정책은 유지한다. 새로운 공식 검증은 `FUTURE_HOLDOUT_VALIDATION.md`에 동결한 500경주 단일 개봉 절차를 따른다.

## 테스트

전체 테스트는 표준 라이브러리 `unittest` discovery로 실행한다.

```powershell
python -m unittest discover -s tests -v
```

검증 범위에는 원본 checksum, 식별자 보존, 경주 무결성, split 격리, 누수 피처 차단, Train 전용 전처리, 확률 합, 일회 Final Test 잠금, bootstrap 반복표본, `no_bet` 정책, 예측 출력 계약이 포함된다. 세부 내용은 [TESTING.md](TESTING.md)를 참고한다.

## 한계

- 데이터는 서울 경주 중심이며 다른 경마장으로 일반화가 확인되지 않았다.
- 현재 모델의 Brier 개선은 bootstrap 95% 신뢰구간에서 확정되지 않았다.
- closing odds는 실제 의사결정 시점에 알 수 없어 경제적 백테스트는 사후 분석이다.
- 실시간 배당 snapshot, 정확한 출발 시각, 배당 변동 및 풀 충격이 없다.
- 현재 정책은 `no_bet`이며 라이브 운영 준비 상태가 아니다.
