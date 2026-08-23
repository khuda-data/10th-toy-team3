# data/processed/ — 중간 산출

이 폴더는 **비어 있는 것이 정상**입니다. `src/0_데이터수집/02_split_by_purpose.py` 가
원천 `data/raw/race_entries.csv.gz` 를 용도별로 갈라 여기에 씁니다.

| 파일 | 무엇인가 | 읽는 코드 |
|---|---|---|
| `model_features.csv` | 모델 학습용 피처 테이블 | `src/1_전처리/05`, `src/2_승률모델/`, `src/3_시장대조/방향실험/` |
| `market_odds.csv` | 배당률·발매금액 등 시장 정보 | `src/2_승률모델/02`·`04`, `src/3_시장대조/방향실험/` |
| `race_outcome.csv` | 착순 등 사후 결과 | `src/2_승률모델/04`, `src/3_시장대조/방향실험/03` |

## 만드는 법

저장소 루트에서:

```bash
python src/0_데이터수집/02_split_by_purpose.py
```

이어서 `src/1_전처리/05_make_versions.py` 가 `data/versions/` 에 전처리 8버전을 만들고,
`07_split_data.py` 가 이를 시간순으로 갈라 **저장소에 올라가 있는** `data/전처리_데이터셋/` 을 만듭니다.

경로는 각 `config.py` 의 `DATA_DIR` · `VERSIONS_DIR` 상수에서 옵니다.
