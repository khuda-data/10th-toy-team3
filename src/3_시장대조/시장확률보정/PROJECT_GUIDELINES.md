# 경주마 시장 확률 보정 프로젝트 가이드라인

이 문서는 본 프로젝트의 데이터 준비, 전처리, 피처 생성, 모델 학습, 확률 보정, 평가, 경제적 백테스트, 예측 출력 및 운영에 적용하는 공식 지침이다.

핵심 변화는 기존의 "배당보다 높은 분류 정확도" 대신 **시장 확률보다 낮은 경주 단위 확률 손실을 달성하는 것**을 1차 목표로 정의하는 것이다.

---

## 1. 프로젝트 목표

### 1.1 공식 목표

> 시간순으로 분리한 미래 경주에서, 모델이 최종 배당으로 계산한 시장 확률보다 낮은 Race Log Loss와 Race Brier Score를 기록하고, 그 개선이 기간·경마장·배당 구간에 걸쳐 재현되는지를 검증한다.

경제적 성과는 별도로 정의한다.

> 모델 확률이 손익분기 확률 `1 / 배당`을 충분히 초과하는 후보만 선택했을 때, 공제율·배당 변동·추정 불확실성을 반영한 순수익률이 양수인지 검증한다.

### 1.2 예측 모델 성공 기준

- 최종 테스트의 `Delta Race Log Loss > 0`
- 경주 단위 bootstrap 95% 신뢰구간 하한이 0 이상
- Race Brier Score가 시장보다 낮음
- 경주별 모델 확률 합이 1
- 개선이 기간별 평가에서도 재현됨
- calibration curve에서 심한 왜곡이 없음
- Test가 모델·피처·임계값 선택에 사용되지 않음

```text
Delta Race Log Loss
    = Market Race Log Loss - Model Race Log Loss
```

- 양수: 모델이 시장보다 우수
- 0: 시장과 동급
- 음수: 시장보다 열세

### 1.3 경제적 성공 기준

- 실제 베팅 가능 시점의 배당을 사용
- `p_model * odds > 1`을 만족
- 안전마진과 배당 변동을 반영
- ROI 95% 신뢰구간을 함께 제시
- 소수 고배당 적중을 제거해도 성과가 유지됨
- 전략 선택에 사용하지 않은 별도 기간에서도 재현됨

예측 우위와 수익 우위는 구분한다.

```text
예측 우위: 모델 확률이 시장 확률보다 실제 결과를 잘 설명함
수익 우위: 모델 확률이 공제율이 포함된 손익분기 확률보다 충분히 높음
```

---

## 2. 프로젝트 범위

### 2.1 1차 범위

- 데이터: 서울 경마
- 기간: 2023-08-05 ~ 2026-08-09
- 관측 단위: 한 경주의 출전마 한 마리
- 기본키: `entry_id`
- 그룹키: `race_id`
- 예측 대상: `win`
- 시장 기준: 단승 배당으로 계산한 `q_market`

부산경남은 2차 확장으로 분리한다. 서울과 부산경남은 각질 데이터 결측 구조가 크게 다르므로 초기 모델에서는 통합하지 않는다.

### 2.2 모델 계층

1. 시장 기준 모델
2. 비시장 정보만 사용하는 독립 모델
3. 시장 확률을 보정하는 시장-offset 모델
4. 경제적 후보를 선택하는 의사결정 모델

시장-offset 모델을 최종 챔피언 후보로 사용한다.

---

## 3. 반드시 먼저 수정할 데이터 문제

### 3.1 `upset_B` 재정의

현재 `upset_B`는 명세와 실제 값이 일치하지 않는다. 기존 열은 신규 학습과 평가에서 사용하지 않는다.

```python
df["longshot"] = (df["pop_pct"] >= 0.50).astype("int8")
df["longshot_win"] = (
    (df["pop_pct"] >= 0.50)
    & (df["win"] == 1)
).astype("int8")
```

이변 모델은 비인기마 후보 안에서 실제 1착을 예측한다.

```python
longshot_df = df[df["pop_pct"] >= 0.50].copy()
y = longshot_df["win"]
```

기존 `upset_A`, `upset_B`, `upset`은 `LEGACY`로 분류하고 사용 금지 목록에 포함한다.

### 3.2 v5~v8 폐기

IQR 이상치 제거 버전은 모델링에 사용하지 않는다.

- 경주의 일부 말만 삭제됨
- 테스트 경주당 평균 출전마가 약 1.16마리만 남음
- 경주 내 경쟁 확률 구조가 파괴됨
- 시간에 따라 제거율이 크게 달라짐

이상치는 행 삭제 대신 다음 방식으로 처리한다.

- Train 기준 winsorization
- `log1p` 변환
- RobustScaler
- 상·하한 clipping
- 이상치 여부 플래그
- 트리 모델에서는 가능한 한 원값 유지

Valid/Test 행은 이상치라는 이유로 삭제하지 않는다.

### 3.3 `fold` 통일

- 원본 `fold`는 `legacy_fold`로 변경
- 새 분할은 코드에서 다시 생성
- 분할 결과를 `split_manifest.csv`로 저장
- 모델은 `split_manifest.csv`만 참조
- `fold`, `legacy_fold`는 피처로 사용하지 않음

---

## 4. 권장 프로젝트 구조

```text
10th-toy-team3-main/
├─ README.md
├─ PROJECT_GUIDELINES.md
├─ requirements.txt
├─ configs/
│  ├─ data_seoul_v2.yaml
│  ├─ features_v2.yaml
│  ├─ model_market.yaml
│  ├─ model_premarket_xgb.yaml
│  ├─ model_market_offset.yaml
│  └─ strategy_value.yaml
├─ data/
│  ├─ raw/final.csv.gz
│  ├─ interim/
│  │  ├─ seoul_entries.parquet
│  │  └─ split_manifest.csv
│  ├─ processed/
│  │  ├─ train.parquet
│  │  ├─ calibration.parquet
│  │  └─ test.parquet
│  └─ manifests/
│     ├─ raw_manifest.json
│     └─ feature_manifest.json
├─ src/
│  ├─ data/
│  │  ├─ load_raw.py
│  │  ├─ validate_schema.py
│  │  ├─ build_splits.py
│  │  └─ validate_races.py
│  ├─ features/
│  │  ├─ registry.py
│  │  ├─ build_market.py
│  │  ├─ build_history.py
│  │  ├─ build_condition.py
│  │  ├─ build_relative.py
│  │  └─ preprocess.py
│  ├─ models/
│  │  ├─ market_baseline.py
│  │  ├─ premarket_logistic.py
│  │  ├─ premarket_xgb.py
│  │  ├─ market_blend.py
│  │  ├─ market_offset.py
│  │  └─ calibrate.py
│  ├─ evaluation/
│  │  ├─ probability_metrics.py
│  │  ├─ race_metrics.py
│  │  ├─ bootstrap.py
│  │  ├─ segment_report.py
│  │  └─ betting_backtest.py
│  ├─ prediction/
│  │  ├─ schema.py
│  │  ├─ predict_race.py
│  │  └─ validate_output.py
│  └─ pipeline/
│     ├─ prepare.py
│     ├─ train.py
│     ├─ evaluate.py
│     └─ predict.py
├─ tests/
│  ├─ test_schema.py
│  ├─ test_race_integrity.py
│  ├─ test_no_leakage.py
│  ├─ test_market_probability.py
│  ├─ test_feature_asof.py
│  └─ test_prediction_sum.py
├─ artifacts/
│  ├─ preprocessors/
│  ├─ models/
│  ├─ calibrators/
│  └─ metadata/
└─ reports/
   ├─ validation/
   ├─ experiments/
   └─ final/
```

---

## 5. 데이터 계약

### 5.1 한 행의 정의

한 행은 특정 경주에 출전한 특정 말 한 마리다.

```text
race_id  = 경주 고유번호
entry_id = race_id + 마번
```

필수 조건:

- `entry_id`는 전체 데이터에서 유일
- `race_id`별 출전마가 2마리 이상
- 정상 경주는 `win=1`이 정확히 한 행
- `q_market`의 경주별 합은 허용오차 내에서 1
- 동일 경주가 서로 다른 fold에 걸치지 않음

### 5.2 컬럼 역할

| 역할 | 용도 | 예시 |
|---|---|---|
| ID | 조인·그룹화 | `race_id`, `entry_id`, `hrNo` |
| SPLIT | 시계열 분할 | `rcDate`, `fold` |
| PRE_RACE | 모델 입력 가능 | `wg`, `rating`, `train_runs_14` |
| MARKET | 시장 기준 또는 보정 | `winOdds`, `q_market` |
| POST_RACE | 모델 입력 금지 | `ord`, `fin_pct`, `win` |
| TARGET | 정답 | `win`, `longshot_win` |
| LEGACY | 호환용·사용 금지 | 기존 `upset_B` |

모든 피처를 registry에 등록한다.

```python
FEATURE_REGISTRY = {
    "train_runs_14": {
        "role": "PRE_RACE",
        "available_at": "before_race",
        "source": "daily_training",
        "missing_policy": "train_median",
    },
    "winOdds": {
        "role": "MARKET",
        "available_at": "market_close",
        "source": "race_result",
        "missing_policy": "reject_race",
    },
    "ord": {
        "role": "POST_RACE",
        "available_at": "after_race",
        "allowed_as_feature": False,
    },
}
```

---

## 6. 데이터 검증

```python
assert df["entry_id"].is_unique
assert df["race_id"].notna().all()
assert df["rcDate"].notna().all()
assert set(df["win"].dropna().unique()) <= {0, 1}
```

경주 단위 검증:

```python
race_check = df.groupby("race_id").agg(
    entries=("entry_id", "size"),
    winners=("win", "sum"),
    q_sum=("q_market", "sum"),
)

assert (race_check["entries"] >= 2).all()
assert (race_check["winners"] == 1).all()
assert ((race_check["q_sum"] - 1).abs() < 1e-6).all()
```

시장 확률 검증:

```python
df["p_raw"] = 1.0 / df["winOdds"]
df["book_sum_check"] = df.groupby("race_id")["p_raw"].transform("sum")
df["q_check"] = df["p_raw"] / df["book_sum_check"]

assert (df["q_market"] - df["q_check"]).abs().max() < 1e-6
```

행 하나에 문제가 있다고 그 행만 제거하지 않는다. 경주 확률을 계산할 수 없는 경우 해당 `race_id` 전체를 제외하고 이유를 기록한다.

---

## 7. 시간순 데이터 분할

### 7.1 최종 고정 분할

- Train: 2023-08-05 ~ 2025-05-11
- Calibration/Validation: 2025-05-17 ~ 2025-12-27
- Final Test: 2025-12-28 ~ 2026-08-09

역할:

- Train: 모델과 전처리기 학습
- Calibration: 확률 보정, 시장 혼합 비율, 베팅 임계값 선택
- Test: 최종 보고서용 1회 평가

Test를 본 뒤 모델이나 피처를 바꾸면 해당 데이터는 더 이상 Test가 아니다.

### 7.2 Train 내부 walk-forward

```text
Fold 1: 과거 60% 학습 -> 다음 10% 검증
Fold 2: 과거 70% 학습 -> 다음 10% 검증
Fold 3: 과거 80% 학습 -> 다음 10% 검증
Fold 4: 과거 90% 학습 -> 다음 구간 검증
```

날짜 경계는 경주일 단위로 맞추며 같은 날짜를 두 fold로 나누지 않는다.

---

## 8. 누수 방지

### 8.1 절대 사용 금지 피처

```text
ord
fin_rank
fin_pct
win
place
resid
upset
upset_A
upset_B
longshot_win
fold
legacy_fold
```

시장 독립 모델에서는 추가로 제외한다.

```text
winOdds
plcOdds
p_raw
book_sum
takeout
q
q_market
logit_q
log_q
pop_rank
pop_pct
is_fav
pl_harville
pl_disc
q_plc
gap_h
gap_d
```

시장 보정 모델에서는 `q_market` 또는 `log(q_market)`만 기본 시장 입력으로 허용한다.

### 8.2 과거 통계 계산

현재 경주 결과를 포함하면 안 된다.

```python
df = df.sort_values(["rcDate", "race_id", "entry_id"])
df["hr_starts"] = df.groupby("hrNo").cumcount()
df["hr_wins_before"] = (
    df.groupby("hrNo")["win"]
      .transform(lambda s: s.shift(1).cumsum())
)
df["hr_winrate"] = (
    df["hr_wins_before"]
    / df["hr_starts"].replace(0, pd.NA)
)
```

동일 날짜에 복수 경주가 가능한 주체는 날짜 단위 snapshot을 사용해 당일 결과가 다음 날짜부터 반영되도록 한다.

---

## 9. 결측치 처리

### 9.1 구조적 결측

예: 데뷔마의 과거 승률, 국6등급의 미부여 rating, 부산경남의 각질 정보.

```python
df["rating_missing"] = df["rating"].isna().astype("int8")
df["rating"] = df["rating"].fillna(0)
```

0이 실제 값으로 가능한 변수는 결측 플래그가 필수다.

### 9.2 랜덤 결측

중앙값은 Train에서만 계산한다.

```python
median = train["hr_rest_days"].median()
train["hr_rest_days"] = train["hr_rest_days"].fillna(median)
valid["hr_rest_days"] = valid["hr_rest_days"].fillna(median)
test["hr_rest_days"] = test["hr_rest_days"].fillna(median)
```

전처리기는 저장하고 예측 시 같은 객체를 사용한다.

---

## 10. 이상치 처리

| 변수 유형 | 권장 처리 |
|---|---|
| 상금·매출액 | `log1p` |
| 휴양일·출전 수 | 상위 99.5% clipping + flag |
| 체중·부담중량 | 원값 유지 또는 Train 기준 winsorization |
| 승률 | Bayesian smoothing |
| z점수·백분위 | 원값 유지 |
| 트리 모델 입력 | 가능한 한 원값 유지 |
| 선형 모델 입력 | RobustScaler 또는 StandardScaler |

```python
lower = train[col].quantile(0.005)
upper = train[col].quantile(0.995)

for frame in [train, valid, test]:
    frame[f"{col}_clipped"] = (
        (frame[col] < lower) | (frame[col] > upper)
    ).astype("int8")
    frame[col] = frame[col].clip(lower, upper)
```

Valid/Test의 분위수를 사용하면 안 된다.

---

## 11. 피처 엔지니어링

### 11.1 우선 사용할 피처

마필 상태 변화:

```text
wg_diff, wgBudam_chg, hr_rest_days, train_days_14,
train_runs_14, train_sec_14, clinic_30d, start_delay
```

과거 경기력:

```text
hr_starts, hr_winrate, hr_plcrate, hr_last_ord,
hr_last_finpct, hr_last_poppct, hr_last_resid,
hr_dist_starts, hr_dist_winrate
```

관계·인적 요소:

```text
jk_winrate, jk_plcrate, tr_winrate, tr_plcrate,
ow_winrate, jkhr_starts, jkhr_winrate, tr_multi
```

경주 조건:

```text
rcDist, rank, ageCond, track, weather,
waterRate, n_run, chaksun1
```

상대적 위치:

```text
wg__z, wg_diff__z, wgBudam__z, rating__z,
train_runs_14__z, age__pr
```

### 11.2 변화량 피처

```python
df["train_runs_change"] = (
    df["train_runs_14"] - df["hr_train_runs_mean_before"]
)
df["jockey_change"] = (
    df["jkNo"] != df.groupby("hrNo")["jkNo"].shift(1)
).astype("int8")
df["distance_change_abs"] = df["hr_dist_chg"].abs()
```

### 11.3 수축 승률

```python
prior = train["win"].mean()
strength = 20
df["jkhr_winrate_smoothed"] = (
    df["jkhr_wins_before"] + strength * prior
) / (
    df["jkhr_starts"] + strength
)
```

`strength`는 Train 내부 walk-forward로 선택한다.

### 11.4 사용을 피할 피처

- 고유값이 많은 이름 문자열
- `hrNo`, `jkNo`, `trNo` 자체
- `birthday`와 `age` 중복 사용
- 거의 동일한 상금 변수 여러 개
- 현재 경주 결과로 계산된 변수
- 예측 시점 이후 확정되는 정보

---

## 12. 시장 기준 모델

```python
def add_market_probability(df):
    df = df.copy()
    df["p_raw"] = 1.0 / df["winOdds"]
    df["book_sum"] = df.groupby("race_id")["p_raw"].transform("sum")
    df["q_market"] = df["p_raw"] / df["book_sum"]
    return df
```

```python
assert np.allclose(
    df.groupby("race_id")["q_market"].sum().values,
    1.0,
)
```

시장 기준 성능은 모델 학습 전에 계산해 고정한다.

---

## 13. 모델 학습 단계

### 13.1 M0: 시장 모델

```text
p = q_market
```

### 13.2 M1: 비시장 Logistic Regression

```python
LogisticRegression(
    penalty="l2",
    C=0.1,
    class_weight=None,
    max_iter=3000,
)
```

`class_weight="balanced"`는 확률을 왜곡할 수 있으므로 기본적으로 사용하지 않는다.

### 13.3 M2: 비시장 XGBoost

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=600,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=10.0,
)
```

탐색 범위:

```text
max_depth: 3, 4, 5, 6
learning_rate: 0.02, 0.03, 0.05
min_child_weight: 10, 20, 40
reg_alpha: 0, 0.5, 1, 2
reg_lambda: 5, 10, 20
subsample: 0.7, 0.8, 1.0
colsample_bytree: 0.7, 0.8, 1.0
```

선택 기준은 ROC-AUC가 아니라 Race Log Loss다. SMOTE, 무작위 undersampling, 임의의 승자 oversampling은 사용하지 않는다.

---

## 14. 모델 확률의 경주 단위 정규화

기본 방식:

```python
df["p_model_race"] = (
    df["p_model_raw"]
    / df.groupby("race_id")["p_model_raw"].transform("sum")
)
```

권장 softmax 방식:

```python
eps = 1e-8
df["model_score"] = np.log(
    np.clip(df["p_model_raw"], eps, 1 - eps)
) - np.log(
    np.clip(1 - df["p_model_raw"], eps, 1 - eps)
)
df["score_exp"] = np.exp(
    df["model_score"]
    - df.groupby("race_id")["model_score"].transform("max")
)
df["p_model_race"] = (
    df["score_exp"]
    / df.groupby("race_id")["score_exp"].transform("sum")
)
```

---

## 15. 권장 최종 모델: 시장 보정

### 15.1 안전한 1차 구현: 기하 혼합

```text
p_i(lambda) proportional to q_i^(1-lambda) * m_i^lambda
```

- `q`: 시장 확률
- `m`: 비시장 모델 확률
- `lambda=0`: 시장만 사용
- `lambda=1`: 모델만 사용

```python
def geometric_blend(df, lam, eps=1e-12):
    log_score = (
        (1 - lam) * np.log(df["q_market"].clip(eps))
        + lam * np.log(df["p_model_race"].clip(eps))
    )
    shifted = log_score - log_score.groupby(df["race_id"]).transform("max")
    numerator = np.exp(shifted)
    return numerator / numerator.groupby(df["race_id"]).transform("sum")
```

```python
lambda_grid = [
    0.00, 0.02, 0.05, 0.10, 0.15,
    0.20, 0.30, 0.50, 0.75, 1.00,
]
```

Calibration Race Log Loss가 가장 낮은 `lambda`를 선택한다. `lambda=0`도 정당한 결과다.

### 15.2 고급 구현: 시장-offset softmax

```text
score_i = log(q_i) + g(X_i)
p_i = softmax_race(score_i)
```

```text
Loss = -sum_r log(p_r,winner) + regularization
```

`g(X)=0`이면 시장과 동일하다. 먼저 선형 `g(X)`로 구현하고 검증 후 비선형으로 확장한다.

---

## 16. 확률 보정

경주 모델에는 temperature scaling을 우선 사용한다.

```text
p_i(T) = softmax(log(p_i) / T)
```

- `T < 1`: 더 날카로운 분포
- `T > 1`: 더 평평한 분포
- `T = 1`: 변화 없음

`T`는 Calibration Race Log Loss를 최소화하도록 선택한다. 보정 데이터와 모델 학습 데이터는 분리한다.

---

## 17. 평가 지표

### 17.1 Race Log Loss

```python
def race_log_loss(df, prob_col):
    winner_prob = df.loc[df["win"] == 1, prob_col].clip(1e-15, 1)
    return -np.log(winner_prob).mean()
```

```python
delta_logloss = (
    race_log_loss(test, "q_market")
    - race_log_loss(test, "p_final")
)
```

### 17.2 Race Brier Score

```python
def race_brier(df, prob_col):
    temp = df.copy()
    temp["sq_error"] = (temp[prob_col] - temp["win"]) ** 2
    return temp.groupby("race_id")["sq_error"].sum().mean()
```

### 17.3 순위 지표

```python
df["pred_rank"] = (
    df.groupby("race_id")["p_final"]
      .rank(ascending=False, method="min")
)
top1_accuracy = (df.loc[df["win"] == 1, "pred_rank"] == 1).mean()
mrr = (1.0 / df.loc[df["win"] == 1, "pred_rank"]).mean()
```

### 17.4 Calibration

확률 구간별 평균 예측확률, 실제 승률, 표본 수, 시장 확률과의 차이를 비교한다. 경주별 합이 1인 `p_final`만 사용한다.

### 17.5 지표 우선순위

1. 시장 대비 Race Log Loss
2. Race Brier Score
3. Calibration
4. 우승마 평균 예측순위와 MRR
5. Top-1 적중률
6. ROI와 Yield

Accuracy와 ROC-AUC는 참고 지표로만 사용한다.

---

## 18. 신뢰구간

출전마 행이 아니라 `race_id` 단위로 bootstrap한다.

```python
race_ids = test["race_id"].unique()

for _ in range(5000):
    sampled = rng.choice(race_ids, size=len(race_ids), replace=True)
    sample = pd.concat([
        test[test["race_id"] == race_id]
        for race_id in sampled
    ])
    delta = (
        race_log_loss(sample, "q_market")
        - race_log_loss(sample, "p_final")
    )
```

평균 Delta Race Log Loss, 95% 신뢰구간, 모델이 우수한 bootstrap 비율을 보고한다.

---

## 19. 경제적 백테스트

### 19.1 손익분기 확률

```python
df["break_even_prob"] = 1.0 / df["winOdds"]
df["expected_edge"] = df["p_final"] * df["winOdds"] - 1
```

`q_market`은 예측 성능 비교용이고 `1 / winOdds`는 수익성 판단용이다.

### 19.2 베팅 조건

Calibration에서만 선택한다.

```text
expected_edge >= 0.05
expected_edge >= 0.10
expected_edge >= 0.15
```

보수적 조건은 `확률 하한 * 배당 > 1.05`로 설정할 수 있다.

### 19.3 동일 금액 베팅

```python
selected = test[test["expected_edge"] >= threshold].copy()
selected["profit"] = np.where(
    selected["win"] == 1,
    selected["winOdds"] - 1,
    -1,
)
roi = selected["profit"].sum() / len(selected)
```

선택 건수, 적중 건수, 평균·중앙 배당, 총 투입금, 총 환급금, ROI, 최대 낙폭, 신뢰구간을 함께 보고한다.

### 19.4 Kelly 비중

```text
kelly = (p * O - 1) / (O - 1)
```

```python
fractional_kelly = max(0, kelly) * 0.10
```

단일 베팅은 자금의 0.5~1% 이하로 제한한다. 주 결과는 동일 금액 백테스트로 보고한다.

---

## 20. 배당 시점

향후 수집할 배당 snapshot:

```text
T-60분
T-30분
T-10분
T-5분
T-3분
T-1분
마감
```

각 예측에는 다음을 기록한다.

```text
prediction_time
odds_snapshot_time
race_start_time
```

실제 전략은 행동 시점에 관측된 배당만 사용한다. 최종 배당은 사후 평가와 closing-line 비교에만 사용한다.

---

## 21. 예측 출력 계약

```text
model_version
prediction_time
odds_snapshot_time
race_id
entry_id
hrNo
hrName
winOdds_snapshot
q_market
p_premarket
p_final
market_delta
break_even_prob
expected_edge
pred_rank
action
```

```text
market_delta = p_final - q_market
expected_edge = p_final * winOdds_snapshot - 1
```

```python
assert output["p_final"].between(0, 1).all()
assert np.allclose(
    output.groupby("race_id")["p_final"].sum(),
    1.0,
    atol=1e-6,
)
assert output["entry_id"].is_unique
```

입력 행 일부가 누락되면 해당 경주 전체를 `prediction_rejected` 처리한다.

---

## 22. 실험 관리

모든 실험은 다음 정보를 저장한다.

```json
{
  "experiment_id": "seoul_offset_xgb_001",
  "data_hash": "...",
  "train_end": "2025-05-11",
  "calibration_end": "2025-12-27",
  "test_end": "2026-08-09",
  "feature_version": "features_v2",
  "market_definition": "normalized_inverse_win_odds",
  "target": "win",
  "model": "xgboost",
  "blend_lambda": 0.10,
  "temperature": 1.15,
  "random_seed": 42
}
```

모든 난수 시드는 고정한다.

```python
SEED = 42
np.random.seed(SEED)
```

| ID | 피처 | 모델 | 시장 사용 | Delta Log Loss | Delta Brier | Top-1 | ROI |
|---|---|---|---|---:|---:|---:|---:|
| M0 | 배당 | 시장 | 기준 | 0 | 0 | 기준 | 기준 |
| M1 | 비시장 | Logistic | 없음 |  |  |  |  |
| M2 | 비시장 | XGBoost | 없음 |  |  |  |  |
| M3 | M2+시장 | 기하혼합 | 혼합 |  |  |  |  |
| M4 | 비시장+시장 | offset | 기준값 |  |  |  |  |

---

## 23. 테스트

최소 자동 테스트:

```text
test_entry_id_unique
test_one_winner_per_race
test_market_probability_sum
test_model_probability_sum
test_no_postrace_features
test_split_has_no_race_overlap
test_split_is_chronological
test_imputer_fitted_on_train_only
test_scaler_fitted_on_train_only
test_history_features_shifted
test_no_rows_removed_from_valid_test
test_prediction_schema
```

```python
for col in selected_features:
    assert FEATURE_REGISTRY[col]["role"] not in {
        "POST_RACE",
        "TARGET",
        "LEGACY",
    }
```

---

## 24. 모델 배포·운영

저장 파일:

```text
preprocessor.joblib
premarket_model.json
blend_config.json
temperature.json
feature_manifest.json
training_metadata.json
```

예측 순서:

```text
1. 입력 schema 검증
2. 경주 전체 출전마 확인
3. 예측 시점 이전 피처만 선택
4. 저장된 전처리기 적용
5. 비시장 모델 점수 생성
6. 경주 단위 확률 정규화
7. 시장 확률과 혼합
8. temperature calibration
9. 경주별 확률 합 검증
10. 시장 차이와 기대값 계산
11. 예측 결과와 모델 버전 기록
```

---

## 25. 운영 모니터링

최근 200~500경주 단위로 추적한다.

- 시장 대비 Delta Race Log Loss
- Delta Race Brier
- Calibration error
- Top-1 적중률
- ROI와 최대 낙폭
- 입력 결측률
- 주요 피처 분포 변화
- 경마장·거리·등급별 성능
- 평균 `market_delta`
- 추천 후보 수

재학습 검토 조건:

- 최근 500경주 Delta Log Loss가 음수
- 주요 피처 결측률이 학습 대비 2배 이상
- calibration이 지속적으로 악화
- 데이터 수집 방식이나 경마 규정 변경
- 일정 기간 신규 데이터 누적

---

## 26. 최종 보고서 구성

1. 연구 질문
2. 시장 기준 확률 정의
3. 데이터 범위와 예측 시점
4. 타깃과 사용 금지 변수
5. 시간순 분할
6. 시장 baseline 성능
7. 독립 모델 성능
8. 시장 보정 모델 성능
9. Delta Race Log Loss 신뢰구간
10. Calibration
11. 기간·경마장·배당별 세그먼트 분석
12. 경제적 백테스트
13. 실패 사례
14. 데이터·모델 한계
15. 재현 방법

결론은 반드시 구분한다.

- 시장보다 확률 예측이 좋은가?
- 시장보다 순위 판별이 좋은가?
- 공제율 이후에도 경제적 우위가 있는가?

하나가 성립한다고 나머지도 성립한다고 표현하면 안 된다.

---

## 27. 권장 개발 순서

1. 원본 CSV를 `data/raw`로 이동하고 checksum 기록
2. 서울 데이터만 추출
3. `upset_B` 오류 수정
4. 경주·시장 확률 무결성 검사
5. 새 시간순 split manifest 생성
6. v5~v8 이상치 제거 파이프라인 폐기
7. 누수 방지 feature registry 작성
8. Train 기준 결측·clipping 전처리 구현
9. M0 시장 baseline 구현
10. M1 Logistic 구현
11. M2 XGBoost 구현
12. 독립 모델 확률의 경주 단위 정규화
13. Calibration에서 기하 혼합 `lambda` 선택
14. Temperature scaling 적용
15. 최종 Test 1회 평가
16. 경주 단위 bootstrap 수행
17. 경제적 백테스트 수행
18. 예측 출력 schema 구현
19. 테스트와 README 작성
20. 최종 보고서 갱신

최우선 마일스톤:

> 현재 시장 확률 `q_market`의 Race Log Loss를 정확히 재현하고, 동일한 테스트 경주에서 시장·독립 모델·시장 혼합 모델을 하나의 평가 코드로 비교할 수 있는 상태를 만든다.

이 기준선이 완성되기 전에는 새로운 모델이나 피처를 무분별하게 추가하지 않는다.

---

## 28. 필수 원칙 요약

1. Accuracy를 최종 모델 선택 기준으로 사용하지 않는다.
2. 시장 대비 Race Log Loss를 1차 지표로 사용한다.
3. 모든 확률은 경주별 합이 1이어야 한다.
4. 한 경주의 일부 말만 임의로 제거하지 않는다.
5. Test로 모델·피처·임계값을 선택하지 않는다.
6. 모든 과거 통계는 현재 경주 결과를 제외하고 계산한다.
7. 전처리기는 Train에만 적합한다.
8. 모델 평가와 베팅 수익 평가를 분리한다.
9. `q_market`과 손익분기 확률 `1 / odds`를 구분한다.
10. 수익성은 표본 수와 신뢰구간을 함께 제시한다.
11. 최종 배당을 실제 예측 시점에 알 수 있었다고 가정하지 않는다.
12. 재현 가능한 설정·데이터 해시·모델 버전을 기록한다.

이 문서는 프로젝트의 공식 기준이며 이후 코드·데이터·보고서는 본 지침과 일치해야 한다.

---

## 29. Top-1 제약 최적화 확장 지침

### 29.1 확장 연구 질문

21단계부터는 기존 확률 예측 목표를 유지하면서 다음 질문을 추가한다.

> 시장보다 Race Log Loss와 Race Brier Score가 나빠지지 않는다는 제약 아래, 새로운 미래 경주에서 시장보다 높은 Top-1 accuracy를 재현할 수 있는가?

Top-1만 높고 확률 품질이 나쁜 모델은 최종 챔피언으로 승격하지 않는다. 반대로 Temperature Scaling처럼 확률값만 바꾸고 경주 내 순서를 바꾸지 않는 방법은 Top-1 개선 방법으로 간주하지 않는다.

```text
Delta Top-1   = Model Top-1 - Market Top-1
Delta LogLoss = Market Race Log Loss - Model Race Log Loss
Delta Brier   = Market Race Brier - Model Race Brier
```

세 개선량은 양수가 모델 우위, 0은 동률, 음수는 모델 열세를 뜻한다.

### 29.2 데이터 사용 경계

- Train은 피처·전처리기·랭킹 모델 학습과 시간순 OOF 예측 생성에 사용한다.
- Calibration은 랭커·확률 변환·시장 결합·게이트 후보 선택에만 사용한다.
- 기존 Final Test(2025-12-28~2026-08-09)는 결과가 이미 공개됐으므로 **설명용 과거 자료**로만 남긴다.
- 기존 Final Test의 행·정답·지표·세그먼트 결과를 피처, 하이퍼파라미터, 게이트, 임계값, lambda 또는 후보 선택에 사용하지 않는다.
- 새 공식 Future Holdout은 2026-08-09 이후의 적격 서울 경주를 시간순으로 500경주 수집해 구성한다.
- Future Holdout의 정답은 후보와 실행 코드가 동결될 때까지 열지 않는다. 정답을 먼저 본 경주는 holdout에서 제외하고 사유를 기록한다.
- 500경주가 모이기 전의 중간 성적을 보고 후보를 바꾸거나 조기 종료하지 않는다.

적격 경주는 완전한 출전 목록, 정확히 한 마리의 우승마, 유효한 단승 배당, 예측 시점 이전 피처를 갖춘 정상 서울 경주다. 취소·무효·공동우승·불완전 수집 경주는 경주 전체를 제외하며 제외 규칙은 결과를 보기 전에 적용한다.

### 29.3 고유 Top-1 판정 규칙

한 경주에서 반드시 한 마리만 1순위로 선택한다.

```text
1차 정렬: p_final 내림차순
2차 동률 해소: q_market 내림차순
3차 동률 해소: entry_id 오름차순
```

시장 기준도 `q_market` 내림차순, `entry_id` 오름차순으로 한 마리를 선택한다. 여러 말에 `rank=1`을 동시에 부여해 적중으로 계산하지 않는다.

### 29.4 모델 후보

모든 독립 후보는 MARKET·POST_RACE·TARGET·LEGACY 피처를 입력으로 사용하지 않는다.

1. `R0`: 시장 `q_market`
2. `R1`: 기존 M2 비시장 XGBoost
3. `R2`: 경주별 그룹을 사용하는 XGBoost pairwise ranker
4. `R3`: 고정 lambda 시장 기하혼합
5. `R4`: 시장 유지/교체 게이트가 경주별 lambda를 선택하는 적응형 기하혼합

기본 랭커는 다음 설정에서 시작한다.

```python
XGBRanker(
    objective="rank:pairwise",
    eval_metric="ndcg@1",
    random_state=42,
)
```

`race_id`를 group으로 전달하고 실제 우승마의 relevance를 1, 나머지를 0으로 둔다. 무작위 행 분할은 금지하며 Train 내부 4-fold expanding walk-forward OOF와 독립 Calibration을 사용한다.

### 29.5 랭킹 점수의 확률 변환

랭킹 점수는 그 자체로 확률이 아니므로 경주별 softmax로 변환한다.

```text
m_ri(T_rank) = softmax_race(score_ri / T_rank)
```

`T_rank`는 Calibration Race Log Loss로만 선택한다. 확률 합은 각 경주에서 1이어야 하고, Temperature Scaling은 순위 개선으로 보고하지 않는다.

### 29.6 시장 유지/교체 게이트

게이트의 기본 행동은 시장 유지다. Calibration에서 충분한 증거가 있는 경주만 랭커의 영향을 허용한다.

```text
lambda_r = 0              if gate_action == keep_market
lambda_r = lambda_switch  if gate_action == trust_ranker

p_ri proportional to q_ri^(1-lambda_r) * m_ri^lambda_r
```

허용 입력은 시장 1·2위 확률 차이, 시장 entropy, 랭커 1·2위 점수 차이, 시장과 랭커의 1위 불일치 여부, 출전두수, 거리·등급·주로 상태, 사전 피처 결측률이다. 게이트 정답과 모든 집계 피처도 현재 경주 결과를 포함하지 않게 생성한다.

### 29.7 후보 선택 순서

후보 선택은 정확도를 단독 최적화하지 않고 다음 순서로 수행한다.

1. 경주별 확률 합, 유일 예측 1위, 누수·시간순·유한값 검사를 통과하지 못한 후보 제거
2. Calibration에서 `Delta LogLoss < 0`인 후보 제거
3. Calibration에서 `Delta Brier < 0`인 후보 제거
4. 통과 후보 중 `Delta Top-1`이 가장 큰 후보 선택
5. Top-1이 같으면 시장을 뒤집은 경주 수가 적은 후보 선택
6. 그래도 같으면 파라미터와 구성요소가 적은 후보 선택

모든 비교는 동일한 경주를 사용한다. 후보 선택 후 피처 목록, 전처리기, 랭커, `T_rank`, 게이트, 임계값, `lambda_switch`, 동률 규칙, seed, 평가 코드와 파일 SHA-256을 동결한다.

### 29.8 성공과 챔피언 승격 기준

개발 구간의 후보 통과는 공식 성공 선언이 아니다. 새 Future Holdout 500경주에서 단 한 번 다음을 평가한다.

Top-1 연구 성공 조건:

- `Delta Top-1 > 0`
- 경주 단위 paired bootstrap 95% 신뢰구간에서 `Delta Top-1` 하한이 0보다 큼
- 시장과 모델의 경주별 정오표에 대한 양측 exact McNemar 검정 `p < 0.05`
- `Delta LogLoss >= 0` 및 `Delta Brier >= 0`
- 모든 경주의 `p_final` 합이 1이고 기간·주요 세그먼트에 치명적 붕괴가 없음

새 챔피언 승격 조건은 위 조건에 더해 기존 공식 기준을 그대로 만족해야 한다.

- `Delta LogLoss > 0`이고 paired race bootstrap 95% 신뢰구간 하한이 0 이상
- `Delta Brier > 0`
- 기간별 개선 재현과 심한 calibration 왜곡 없음

Top-1 연구 성공만 충족하고 챔피언 승격 조건을 충족하지 못하면 `research_challenger`로만 저장한다. 모든 조건을 충족하지 못하면 기존 챔피언과 `no_bet` 정책을 유지하고 결과를 `no_change`로 기록한다. 경제적 우위는 별도 백테스트 없이는 주장하지 않는다.

통계 검정은 `race_id` 단위 paired bootstrap 10,000회, seed 42를 기본값으로 사용한다. Future Holdout을 연 후 검정법·반복 수·유의수준을 변경하지 않는다.

### 29.9 21~29단계 권장 개발 순서

21. 본 확장 목표·데이터 경계·승격 기준을 문서와 manifest로 동결
22. Train OOF와 Calibration에서 시장/모델 1위 불일치 경주 분석
23. 경주 그룹 랭킹 데이터셋·무결성 검사 구현
24. 시간순 4-fold walk-forward 랭커 학습과 후보 비교
25. 랭킹 점수의 경주 확률 변환과 Calibration temperature 선택
26. 시장 유지/교체 게이트와 경주별 lambda 구현
27. 비열화 제약 아래 후보 선택 및 사전 동결
28. 새 Future Holdout 500경주에서 1회 평가와 paired 검정
29. 예측 계약·테스트·README·최종 보고서 갱신

21~27단계에서 기존 Final Test 성능을 근거로 설계를 바꾸지 않는다. 28단계에 필요한 새 Future Holdout이 아직 500경주에 도달하지 않았다면 작업 상태는 실패가 아니라 `pending_data`다.
