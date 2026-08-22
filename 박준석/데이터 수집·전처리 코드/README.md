# 데이터 수집·전처리 코드

프로젝트의 가장 첫 단계 — 한국마사회 공공데이터를 API로 모아 `final.csv`를 만들고, 그걸 다시 용도별 파일로 정리하는 코드입니다.

## 코드 설명

| 파일 | 설명 | 작성자 |
|---|---|---|
| `kra_client.py` | 한국마사회 공공데이터포털 API 공통 클라이언트. RC경마경주정보·기수 성적·경주마 성적 등 여러 API를 동일한 인터페이스로 호출 | 박준석 (junseok) |
| `collect_rc_race.py` | 서울경마장 경주 데이터(배당률·마체중·트랙상태·날씨·착순 등)를 API로 수집해 CSV로 저장. 서비스 키는 `.env`의 `KRA_SERVICE_KEY`에서 읽음 | 박준석 (junseok) |
| `preprocess_final.py` | 팀원이 만든 `final.csv`(156컬럼·56,648행)를 모델 학습용/배당률 분석용/사후 결과용 세 파일로 분리 | 박준석 (junseok) |
| `config.py` | 컬럼 분류·유틸리티 함수 등 파이프라인 공통 설정 모듈 (팀 공용 `src/pipeline/config.py`를 이 파이프라인에서 그대로 import해 사용) | 팀 공용 (원 작성자 미상) |

## 실행 순서

`kra_client.py`는 `collect_rc_race.py`가 사용하는 라이브러리 모듈이라 직접 실행하지 않습니다. `collect_rc_race.py`로 원본 데이터를 모으고, 팀원들의 결과물을 합쳐 `final.csv`가 만들어진 뒤 `preprocess_final.py`를 실행하는 순서입니다.
