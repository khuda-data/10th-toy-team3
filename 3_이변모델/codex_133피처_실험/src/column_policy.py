"""피처로 쓰면 안 되는 열 정책 — 누수(leak) 열과 식별자 열.

원래 이 상수들은 `src/training/train_ml.py` 상단에 있었다. 그런데 train_ml.py는
최종보고서 1장에서 **종결된 접근**(EDGE = p_model - p_market 구조)의 학습 스크립트라
`archive/`로 내려가야 하는데, 현행 핵심 스크립트인
`benter_market_anchored_walkforward.py`가 이 두 상수를 import하고 있었다.
"죽은 스크립트를 상수 때문에 살려두는" 상태였으므로 공용 모듈로 분리했다.

LEAK_COLS — 경주 결과 이후에만 알 수 있거나 시장가 그 자체인 열.
  펀더멘털 모델의 피처 후보에서 반드시 제외한다. 두 종류가 섞여 있다:
    (a) 사후정보: ord, fin_rank, win, place, resid, upset_* 등
    (b) 시장정보: winOdds, plcOdds, q, logit_q, pop_rank, is_fav 등
  (b)를 제외하는 이유는 누수라서가 아니라, 이 프로젝트의 설계상 시장가는
  **피처가 아니라 ln(q) 오프셋(계수 1 고정)** 으로 들어가기 때문이다. 피처로도
  넣으면 같은 정보가 이중 계상된다.
  주의: hr_resid/jk_resid/hr_last_poppct 등 '과거 시장 판단에서 파생된' 열도
  여기 포함돼 있다 — 현재 경주의 시장가는 아니지만 시장 독립적이지도 않다.

ID_COLS — 식별자/라벨 열. 숫자로 저장돼 있어도 크기 비교가 의미 없으므로
  수치 피처로 쓰면 안 된다(hrNo, jkNo, trNo, owNo가 대표적).
"""
from __future__ import annotations

LEAK_COLS = {
    # (a) 사후정보 — 경주가 끝나야 알 수 있는 값
    "ord", "fin_rank", "fin_pct", "win", "place", "resid",
    "upset", "upset_A", "upset_B",
    "chaksun1", "chaksun2", "chaksun3", "chaksun4", "chaksun5",
    # (b) 시장정보 — 오프셋으로만 쓰고 피처로는 쓰지 않는다
    "winOdds", "plcOdds", "p_raw", "book_sum", "takeout", "q", "logit_q",
    "log_q", "pop_rank", "pop_pct", "is_fav", "q_plc", "gap_h", "gap_d",
    "pl_harville", "pl_disc",
    # (c) 과거 시장 판단에서 파생된 열 — 완전한 시장독립이 아니다
    "hr_resid", "jk_resid", "tr_resid", "ow_resid",
    "hr_last_resid", "hr_last_poppct", "hr_resid__z", "hr_resid__pr",
    # (d) 최종 발매 풀 — 경주 마감 뒤 확정되는 시장 유동성 정보
    "winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse",
}

ID_COLS = {
    "hrName", "jkName", "trName", "owName", "name", "rcName", "birthday",
    "race_id", "entry_id", "meet", "meet_cd", "rcDate", "rcDay", "rcNo",
    "fold", "tool_set", "prizeCond", "hrNo", "jkNo", "trNo", "owNo",
}

__all__ = ["LEAK_COLS", "ID_COLS"]
