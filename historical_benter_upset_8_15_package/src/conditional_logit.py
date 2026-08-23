"""경주 단위 조건부 로짓(conditional / multinomial logit) — 경마 확률모형의 표준 도구.

왜 필요한가:
  기존 프로젝트는 "이 말이 1등 할 확률"을 **이진 분류기**로 학습한 뒤 경주 안에서
  사후적으로 sum-to-1 정규화(race_normalize)했다. 하지만 경마는 본질적으로
  "한 경주 안 N마리 중 정확히 1마리가 이기는" 이산선택(discrete choice) 문제다.
  조건부 로짓은 그 구조를 **손실함수 자체에** 넣는다:

      P(말 i가 경주 j에서 1등) = exp(x_i·β) / Σ_{k∈경주 j} exp(x_k·β)

  이진 분류 + 사후 정규화보다 통계적으로 훨씬 효율적이고(같은 데이터로 더 좁은
  신뢰구간), 무엇보다 **계수의 표준오차/우도비 검정**을 쓸 수 있다. 이게 결정적인
  이유는 아래 statistical power 문제 때문이다.

검정력(statistical power) 문제:
  기존 EDGE 백테스트는 "경주당 1마리 × 상위 10~30%"로 걸러서 베팅 64~276건으로
  ROI 부트스트랩 CI를 만들었다. 단승 적중률이 ~10%, 배당 분산이 극단적이라
  이 표본으로는 진짜 +5% edge가 있어도 CI가 절대 0을 배제하지 못한다
  (= 검정력이 사실상 없다). 반면 조건부 로짓 우도비 검정은 **전체 5,361경주**를
  전부 쓰므로 훨씬 작은 신호도 잡아낸다. "edge가 없다"와 "표본이 부족해서 안 보인다"를
  구분하려면 반드시 이쪽 검정이 필요하다.

참고문헌: Bill Benter (1994) "Computer Based Horse Race Handicapping and Wagering
Systems: A Report"; Bolton & Chapman (1986) Management Science 32(8).
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats


@dataclass
class RaceGroups:
    """경주 단위로 정렬된 데이터의 그룹 구조.

    entry는 반드시 race 순서대로 정렬돼 있어야 하며(같은 경주끼리 연속),
    offsets는 np.add.reduceat용 시작 인덱스다.
    """

    offsets: np.ndarray       # (n_races,)  각 경주의 시작 행 인덱스
    sizes: np.ndarray         # (n_races,)  각 경주의 출전 두수
    winner_rows: np.ndarray   # (n_races,)  각 경주 1등 말의 행 인덱스
    n_rows: int
    n_races: int

    @classmethod
    def from_sorted(cls, race_codes: np.ndarray, win: np.ndarray) -> "RaceGroups":
        """race_codes는 정렬된 정수 코드(같은 경주끼리 연속), win은 0/1."""
        if np.any(np.diff(race_codes) < 0):
            raise ValueError("race_codes가 정렬돼 있지 않습니다 — 먼저 정렬하세요.")
        boundaries = np.r_[0, np.flatnonzero(np.diff(race_codes)) + 1]
        sizes = np.diff(np.r_[boundaries, len(race_codes)])
        winners = np.flatnonzero(win == 1)
        # 경주당 정확히 1마리 승자여야 한다.
        winner_race = np.searchsorted(boundaries, winners, side="right") - 1
        if len(np.unique(winner_race)) != len(boundaries) or len(winners) != len(boundaries):
            raise ValueError(
                f"경주당 승자가 정확히 1마리가 아닙니다 "
                f"(races={len(boundaries)}, winners={len(winners)}). 먼저 필터링하세요."
            )
        order = np.argsort(winner_race, kind="stable")
        return cls(offsets=boundaries, sizes=sizes, winner_rows=winners[order],
                   n_rows=len(race_codes), n_races=len(boundaries))


def group_logsumexp(eta: np.ndarray, g: RaceGroups) -> np.ndarray:
    """경주별 logsumexp — 수치적으로 안전하게(최댓값 차감)."""
    m = np.maximum.reduceat(eta, g.offsets)
    shifted = np.exp(eta - np.repeat(m, g.sizes))
    s = np.add.reduceat(shifted, g.offsets)
    return m + np.log(s)


def group_softmax(eta: np.ndarray, g: RaceGroups) -> np.ndarray:
    """경주별 softmax — 경주 안에서 확률 합이 1이 되도록."""
    lse = group_logsumexp(eta, g)
    return np.exp(eta - np.repeat(lse, g.sizes))


@dataclass
class ClogitFit:
    beta: np.ndarray
    names: list[str]
    loglik: float
    loglik_null: float          # 모든 말이 동확률(1/두수)일 때의 로그우도
    n_races: int
    n_rows: int
    std_errors: np.ndarray | None = None
    converged: bool = True
    offset_name: str | None = None
    _cov: np.ndarray | None = field(default=None, repr=False)

    @property
    def mcfadden_r2(self) -> float:
        """McFadden pseudo-R². Benter가 모형 유용성 지표로 쓴 값과 같은 계열."""
        return 1.0 - self.loglik / self.loglik_null

    @property
    def z_values(self) -> np.ndarray | None:
        if self.std_errors is None:
            return None
        return self.beta / self.std_errors

    @property
    def p_values(self) -> np.ndarray | None:
        z = self.z_values
        if z is None:
            return None
        return 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))

    def summary_rows(self) -> list[dict]:
        se = self.std_errors if self.std_errors is not None else np.full(len(self.beta), np.nan)
        z = self.z_values if self.z_values is not None else np.full(len(self.beta), np.nan)
        p = self.p_values if self.p_values is not None else np.full(len(self.beta), np.nan)
        return [
            {"term": n, "coef": float(b), "std_err": float(s), "z": float(zz), "p_value": float(pp)}
            for n, b, s, zz, pp in zip(self.names, self.beta, se, z, p)
        ]


def _nll_and_grad(beta, X, g, offset, l2=0.0):
    eta = X @ beta
    if offset is not None:
        eta = eta + offset
    lse = group_logsumexp(eta, g)
    ll = eta[g.winner_rows].sum() - lse.sum()
    p = np.exp(eta - np.repeat(lse, g.sizes))
    # grad(LL) = Σ_j x_winner - Σ_j Σ_i p_i x_i
    grad_ll = X[g.winner_rows].sum(axis=0) - (p[:, None] * X).sum(axis=0)
    if l2 > 0.0:
        # L2(ridge) 벌점 — 피처가 많고 경주 수가 적을 때 과적합을 억제한다.
        ll = ll - 0.5 * l2 * float(beta @ beta)
        grad_ll = grad_ll - l2 * beta
    return -ll, -grad_ll


def _hessian(beta, X, g, offset):
    """관측 정보행렬 -d²LL/dβ² = Σ_j [ Σ_i p_i x_i x_i' - (Σ_i p_i x_i)(Σ_i p_i x_i)' ]"""
    eta = X @ beta
    if offset is not None:
        eta = eta + offset
    p = group_softmax(eta, g)
    px = p[:, None] * X
    # Σ_i p_i x_i x_i'  (전체 합)
    term1 = X.T @ px
    # 경주별 Σ_i p_i x_i
    mean_x = np.add.reduceat(px, g.offsets, axis=0)      # (n_races, k)
    term2 = mean_x.T @ mean_x
    return term1 - term2


def fit_conditional_logit(
    X: np.ndarray,
    g: RaceGroups,
    names: list[str],
    offset: np.ndarray | None = None,
    offset_name: str | None = None,
    compute_se: bool = True,
    l2: float = 0.0,
) -> ClogitFit:
    """경주 단위 조건부 로짓 최대우도 적합.

    offset을 주면 그 항의 계수를 1로 **고정**한 채 나머지 계수를 추정한다.
    ln(시장확률)을 offset으로 넣으면 "시장가를 그대로 인정하고, 그 위에 남는
    잔차 정보가 있는가"를 검정하는 모형이 된다 — 시장이 이미 반영한 정보로
    가짜 edge가 생기는 걸 구조적으로 막아준다.
    """
    X = np.ascontiguousarray(X, dtype=float)
    k = X.shape[1]
    res = optimize.minimize(
        _nll_and_grad, np.zeros(k), args=(X, g, offset, l2), jac=True,
        method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )
    beta = res.x
    # 보고되는 loglik은 벌점을 제외한 순수 로그우도여야 우도비 검정이 성립한다.
    loglik = loglik_of(beta, X, g, offset)
    # null: 경주 안 모든 말이 동확률 -> LL = Σ_j -ln(두수)
    loglik_null = float(-np.log(g.sizes).sum())

    se, cov = None, None
    if compute_se:
        H = _hessian(beta, X, g, offset)
        try:
            cov = np.linalg.inv(H)
            diag = np.diag(cov)
            se = np.sqrt(np.where(diag > 0, diag, np.nan))
        except np.linalg.LinAlgError:
            se, cov = None, None

    return ClogitFit(beta=beta, names=list(names), loglik=loglik, loglik_null=loglik_null,
                     n_races=g.n_races, n_rows=g.n_rows, std_errors=se,
                     converged=bool(res.success), offset_name=offset_name, _cov=cov)


def loglik_of(beta: np.ndarray, X: np.ndarray, g: RaceGroups,
              offset: np.ndarray | None = None) -> float:
    """주어진 계수로 (보통 다른 표본에서) 로그우도를 계산 — 표본외 평가용."""
    eta = X @ beta
    if offset is not None:
        eta = eta + offset
    lse = group_logsumexp(eta, g)
    return float(eta[g.winner_rows].sum() - lse.sum())


def mcfadden_r2_of(beta: np.ndarray, X: np.ndarray, g: RaceGroups,
                   offset: np.ndarray | None = None) -> float:
    ll = loglik_of(beta, X, g, offset)
    ll0 = float(-np.log(g.sizes).sum())
    return 1.0 - ll / ll0


def likelihood_ratio_test(ll_full: float, ll_restricted: float, df: int) -> dict:
    """중첩모형 우도비 검정: 2(LL_full - LL_restricted) ~ chi2(df).

    "제한모형(예: 시장확률만)에 비해 완전모형(시장확률+펀더멘털)이 유의하게
    나은가"를 전체 경주를 다 써서 검정한다 — 베팅 몇십 건짜리 ROI CI보다
    검정력이 압도적으로 높다.
    """
    stat = 2.0 * (ll_full - ll_restricted)
    p = float(stats.chi2.sf(stat, df)) if stat > 0 else 1.0
    return {"lr_statistic": float(stat), "df": int(df), "p_value": p,
            "significant_5pct": bool(p < 0.05)}
