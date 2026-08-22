# 20단계 완료 요약 — 최종 결과보고서

## 산출물

- 최종 보고서: `경주마_시장확률_보정_최종_결과보고서.docx`
- 생성 스크립트: `scripts/build_final_report.py`
- 원본 1차 보고서: 변경 없이 보존
- 최종 보고서 크기: 104,608 bytes
- SHA-256: `a5997eb8820456b66efed1c7f0efce0da27946723af7a137e477195946725a83`

## 최종 결론

- 동결 M2 모델은 Final Test에서 시장보다 낮은 경주 Log Loss와 Brier score를 기록했다.
- 시장 대비 Log Loss 개선량은 `+0.003902`이며 95% paired race bootstrap CI는 `[+0.001085, +0.006721]`이다.
- Brier 개선량은 `+0.000602`이나 95% CI가 0을 포함하므로, 두 핵심 지표 모두에서 안정적 우위를 요구한 엄격한 성공 기준은 미충족이다.
- Top-1 accuracy는 M2 `37.48%`, 시장 `37.80%`로 시장이 `0.315%p` 높다.
- Calibration과 Final Test의 최대 기대우위가 모두 음수이고 5%·10%·15% 임계값 선택이 0건이므로 공식 행동은 `no_bet`이다.
- 모델은 연구용 확률 보정기로 유지하며 실시간 운영은 실제 사전 배당 스냅샷·공식 시작시각·완전한 출전 목록·드리프트 모니터링이 확보될 때까지 차단한다.

## 문서 검증

- Microsoft Word 렌더링 기준 14쪽 전체를 PNG로 육안 검수했다.
- 빈 페이지, 잘린 문장·표·그림, 깨진 한글이 없음을 확인했다.
- 접근성 감사: high 0, medium 0, low 0.
- 제목 구조: Heading 1 12개, Heading 2 12개.
- 모든 표의 `tblW`, `tblInd`, `tblGrid`, `tcW`가 9,360 DXA 고정 폭 계약과 일치한다.
- 그림 2개에 대체 텍스트를 포함했다.
- 프로젝트 가상환경에서 회귀 테스트 72개가 모두 통과했다.

## 재현

```powershell
python scripts/build_final_report.py
python -m unittest discover -s tests -v
```
