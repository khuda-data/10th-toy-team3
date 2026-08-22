# 개발 환경 설정

## 권장 환경

- Python 3.12
- Windows PowerShell 또는 일반 터미널
- 프로젝트 루트에서 명령 실행

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

PowerShell 실행 정책 때문에 활성화가 막히면 가상환경의 Python을 직접 사용할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

## 설치 확인

```powershell
python -c "import pandas, sklearn, xgboost; print(pandas.__version__, sklearn.__version__, xgboost.__version__)"
```

프로젝트 고정 버전은 `requirements.txt`를 기준으로 한다.

## 데이터 확인

다음 파일이 있어야 한다.

```text
data/raw/final.csv.gz
data/interim/seoul_entries.csv.gz
data/interim/split_manifest.csv
```

원본과 manifest를 검증한다.

```powershell
python -m src.data.validate_schema
```

## 문제 해결

### `ModuleNotFoundError`

현재 Python과 패키지를 설치한 Python이 같은지 확인한다.

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

### XGBoost DLL 오류

64비트 Python 3.12와 최신 Microsoft Visual C++ Runtime을 사용한다. 가상환경을 새로 만든 뒤 `requirements.txt`를 다시 설치한다.

### 메모리 부족

M2 walk-forward는 여러 XGBoost 모델을 순차 학습한다. 다른 대용량 프로그램을 닫고 `n_jobs`를 제한해 실행할 수 있지만, 설정을 변경한 결과는 기존 실험과 별도 버전으로 기록해야 한다.

### Final Test 명령이 실패함

`pre_final_test_freeze.json` 또는 `stage_15_final_test.json`이 이미 있으면 재실행 방지 장치가 동작한 것이다. 현재 저장소에서는 정상적인 상태다. 기존 결과를 삭제하거나 덮어쓰지 않는다.
