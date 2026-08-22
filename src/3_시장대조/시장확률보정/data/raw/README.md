# data/raw — 원본 데이터를 여기에 두세요

이 폴더는 **비어 있습니다.** 원본 파일이 저장소 다른 곳에 이미 한 벌 있어서, 같은 22MB를 두 번 담지 않으려고 비워뒀습니다.

## 무엇이 필요한가

```
data/raw/final.csv.gz
```

`src/data/load_raw.py`의 `DEFAULT_RAW_PATH`가 가리키는 경로입니다. 아래 코드와 테스트가 이 파일을 읽습니다.

- `src/data/load_raw.py` — 원본 로더
- `src/data/build_seoul_interim.py` — 서울 필터링, SHA-256 기록
- `src/data/validate_schema.py` — 스키마 검증
- `tests/test_schema.py` — 파일 존재와 체크섬 확인

## 어떻게 채우는가

저장소 최상위의 [`data/race_entries.csv.gz`](../../../../../data/race_entries.csv.gz)가 바로 그 파일입니다. 이름만 다릅니다 — 정리 과정에서 "final"이 단계를 뜻하는 것처럼 읽혀 `race_entries`로 바꿨습니다.

저장소 루트에서 실행하세요.

```powershell
Copy-Item "data\race_entries.csv.gz" `
          "src\3_시장대조\시장확률보정\data\raw\final.csv.gz"
```

```bash
cp data/race_entries.csv.gz \
   src/3_시장대조/시장확률보정/data/raw/final.csv.gz
```

56,648 출전행 × 156열 · 서울과 부경을 모두 포함한 원천 테이블입니다. 이 파이프라인은 여기서 서울만 걸러 `data/interim/seoul_entries.csv.gz`(32,888행)를 만듭니다.

## 채우지 않으면

`data/interim/`의 중간 데이터가 이미 들어 있어서 **대부분의 코드는 그대로 돕니다.** 다만 원본이 필요한 아래 항목은 실패합니다.

| 항목 | 결과 |
|---|---|
| `tests/test_schema.py` | 파일이 없어 실패 |
| `python -m src.data.build_seoul_interim` | 중간 데이터 재생성 불가 |
| `python -m src.data.validate_schema` | 스키마 검증 불가 |

나머지 테스트 17개와 학습·평가 스크립트는 중간 데이터만으로 동작합니다.

---

이 폴더가 비어 있는 것은 원작자의 의도이기도 합니다. `INDEX.md`에 "제외한 것: `.venv`, `.git`, `__pycache__`, 전처리 데이터셋 CSV, `final.csv` 원본"이라고 적혀 있습니다.
