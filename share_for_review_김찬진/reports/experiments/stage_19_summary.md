# 개발 19단계 결과 — 테스트와 README

생성 시점: 2026-08-18

## 완료 항목

- 루트 `README.md`를 프로젝트 목표, 데이터, 모델, 평가 결과, 재현 명령 및 운영 한계를 포함하도록 전면 작성
- `SETUP.md`에 Windows·macOS·Linux 개발 환경 구성과 문제 해결 절차 작성
- `TESTING.md`에 전체·파일별 테스트 명령과 각 테스트 계약 작성
- `scripts/run_tests.ps1` Windows 테스트 진입점 추가
- `requirements.txt`의 검증 Python 환경을 현재 환경에 맞게 수정
- 문서와 문서화된 결과 경로를 확인하는 자동 테스트 추가
- 문서·테스트 파일 SHA-256 manifest 생성

## 문서에 고정한 핵심 결론

```text
Final Test에서 M2는 시장보다 낮은 Race Log Loss와 Brier를 기록했다.
Log Loss 개선은 bootstrap에서 지지되지만 Brier 개선은 불확실하다.
공제율을 넘는 양의 기대수익 후보가 없어 운영 정책은 no_bet이다.
현재 시스템은 라이브 운영 준비 상태가 아니다.
```

## 재현 명령

```powershell
.\scripts\run_tests.ps1
```

또는:

```powershell
python -m unittest discover -s tests -v
```

Final Test 평가 명령은 일회성 잠금 명령이며 일반 테스트 절차에 포함하지 않았다.

다음 개발 단계는 20단계 최종 보고서 갱신이다.
