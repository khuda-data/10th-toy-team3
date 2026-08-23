# data/ — 데이터 계층

원본에서 학습 가능한 데이터셋까지 **네 단계**를 거칩니다. 저장소에는 **양 끝만** 들어 있습니다.

```
raw/                 ← 저장소에 있음 · 원천 (56,648행 × 156열)
  └ race_entries.csv.gz
       │  src/0_데이터수집/02_split_by_purpose.py
       ▼
processed/           ← 비어 있음 · 용도별 분리 (model_features · market_odds · race_outcome)
       │  src/1_전처리/05_make_versions.py
       ▼
versions/            ← 비어 있음 · 전처리 8버전 (이상치 2종 × 스케일링 4종)
       │  src/1_전처리/07_split_data.py
       ▼
전처리_데이터셋/       ← 저장소에 있음 · v1~v8 × train/valid/test
```

가운데 두 단계를 뺀 것은 **코드가 다시 만들어내기 때문**입니다. 되살리는 명령은
[`processed/README.md`](processed/README.md)에 있습니다.

## 바로 쓰려면

전처리가 끝난 `전처리_데이터셋/`을 그대로 읽으면 됩니다. 중간 단계를 돌릴 필요가 없습니다.

```python
import pandas as pd

VERSION = "v1_base"          # 트리 모델 (RF · XGBoost · LightGBM)
# VERSION = "v2_standard"    # 로지스틱 회귀 · SVM

BASE = f"data/전처리_데이터셋/{VERSION}"
dtypes = {"hrNo": str, "jkNo": str, "trNo": str, "owNo": str}

train = pd.read_csv(f"{BASE}/train.csv.gz", encoding="utf-8-sig", dtype=dtypes)
valid = pd.read_csv(f"{BASE}/valid.csv.gz", encoding="utf-8-sig", dtype=dtypes)
test  = pd.read_csv(f"{BASE}/test.csv.gz",  encoding="utf-8-sig", dtype=dtypes)
```

**v5~v8은 신규 학습에 쓰면 안 됩니다.** 이유는 [`전처리_데이터셋/사용정책.md`](전처리_데이터셋/사용정책.md)에,
왜 시간순으로 잘랐는지는 [`전처리_데이터셋/분할안내.md`](전처리_데이터셋/분할안내.md)에 있습니다.

## 코드에서 경로 쓰기

경로를 문자열로 적지 말고 `config.py` 상수를 쓰세요. 실행 위치와 무관하게 같은 곳을 가리킵니다.

| 상수 | 가리키는 곳 |
|---|---|
| `RAW_ENTRIES` | `data/raw/race_entries.csv.gz` |
| `DATA_DIR` | `data/processed/` |
| `VERSIONS_DIR` | `data/versions/` |
| `SPLITS_DIR` | `data/전처리_데이터셋/` |

열 156개의 뜻과 지켜야 할 제약은 [`../docs/데이터.md`](../docs/데이터.md)에 있습니다.
