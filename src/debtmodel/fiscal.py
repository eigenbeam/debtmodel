"""Debt/GDP dynamics with an explicit maturity ladder.

The textbook identity for the debt ratio is

    delta_b = (r - g) * b - s

with ``r`` the real interest rate, ``g`` real growth and ``s`` the primary
surplus, both as shares of GDP.  It is a useful summary and a poor projection
tool, because it assumes the entire stock of debt instantly bears the current
market rate.  Real debt reprices only as it matures, and the US stock reprices
unusually fast -- about 40% within a year, 64% within three.

This module keeps the same accounting but carries a **maturity ladder**: how
much principal comes due in each future year and at what interest rate it was
issued.  Each period the maturing slice is refinanced at whatever the market
then charges, the rest keeps its old coupon, and the effective rate on the
whole stock drifts from its historical level toward the market level at a
speed the ladder determines.  That makes the passthrough from market rates to
the budget an output rather than an assumption, and it lets issuance policy
(how long Treasury borrows) be a scenario lever.

Everything is nominal.  Inflation therefore appears in two places at once --
raising nominal GDP in the denominator and raising nominal interest on new
issuance in the numerator -- which is the honest way to show why anticipated
inflation is close to a wash for the trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["DebtLadder", "Assumptions", "Projection", "project"]

_MAX_MATURITY = 30


def _as_path(x: float | np.ndarray, years: int) -> np.ndarray:
    """Broadcast a scalar to a path, or check the length of a given path."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return np.full(years, float(a))
    if a.size == years:
        return a.astype(float)
    if a.size < years:  # extend with the final value
        return np.concatenate([a, np.full(years - a.size, a[-1])])
    return a[:years].astype(float)


@dataclass(frozen=True)
class DebtLadder:
    """Principal outstanding by remaining maturity, and its average coupon.

    ``principal[i]`` matures in ``i + 1`` years and pays ``coupon[i]`` on its
    face each year until then.  Bills are carried as one-year principal with a
    coupon equal to their yield, which is how they hit the budget even though
    they are legally discount instruments.
    """

    principal: np.ndarray
    coupon: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.principal, dtype=float)
        c = np.asarray(self.coupon, dtype=float)
        if p.shape != c.shape:
            raise ValueError(f"principal {p.shape} and coupon {c.shape} must match")
        if np.any(p < 0):
            raise ValueError("principal must be non-negative")
        object.__setattr__(self, "principal", p)
        object.__setattr__(self, "coupon", c)

    @property
    def total(self) -> float:
        return float(self.principal.sum())

    @property
    def effective_rate(self) -> float:
        """Weighted average coupon: interest cost divided by debt outstanding."""
        return float((self.principal * self.coupon).sum() / self.principal.sum())

    @property
    def average_maturity(self) -> float:
        years = np.arange(1, self.principal.size + 1)
        return float((years * self.principal).sum() / self.principal.sum())

    def rescaled(self, total: float, effective_rate: float | None = None) -> "DebtLadder":
        """Same shape, scaled to a given stock and (optionally) average rate.

        Rescaling the coupon preserves the *pattern* of high- and low-coupon
        vintages across the ladder while pinning the average to an observed
        interest-cost-to-debt ratio.
        """
        p = self.principal * (total / self.total)
        c = self.coupon
        if effective_rate is not None:
            c = c * (effective_rate / self.effective_rate)
        return DebtLadder(p, c)


@dataclass(frozen=True)
class Assumptions:
    """A scenario: paths for the economy, the budget and debt management.

    Scalars are held constant over the projection; arrays are read year by year
    and extended with their final value if short.  Rates are decimals.
    """

    #: Real GDP growth.
    real_growth: float | np.ndarray = 0.018
    #: Inflation (GDP deflator).  Enters nominal GDP and nominal rates alike.
    inflation: float | np.ndarray = 0.023
    #: Real interest rate on newly issued short-dated debt -- what actually
    #: happens.
    real_short_rate: float | np.ndarray = 0.017

    #: What the market *expects* the real short rate to be, used to price
    #: long-dated issuance.  ``None`` means expectations are correct (perfect
    #: foresight), which is the right default for a scenario meant as a
    #: forecast.
    #:
    #: Setting this to the baseline path while ``real_short_rate`` follows a
    #: spike models an *unanticipated* tightening, and only then does the
    #: maturity of the debt matter in the way the debate assumes.  Long debt
    #: does not protect against rate rises everyone already sees coming --
    #: those are priced into the coupon on the day it is issued.  It protects
    #: against surprises.
    expected_real_short_rate: float | np.ndarray | None = None

    #: Likewise for inflation: what the market prices in when buying bonds.
    expected_inflation: float | np.ndarray | None = None
    #: Primary balance as a share of GDP.  Positive is a surplus.
    primary_balance: float | np.ndarray = -0.025

    #: Share of gross issuance placed at each maturity, indexed 1..30 years.
    #: Defaults to something close to current Treasury practice: heavy bills,
    #: a large belly, a thin long end.
    issuance_shares: np.ndarray = field(
        default_factory=lambda: _default_issuance()
    )

    #: Extra yield demanded per 10 percentage points of debt/GDP above the
    #: starting ratio.  Zero means investors never charge for a bigger debt;
    #: a positive value makes the trajectory self-reinforcing.
    risk_premium_per_10pp: float = 0.0

    #: Basis points shaved off the rate on *new* issuance by financial
    #: repression -- captive demand, regulatory requirements, forced holdings.
    repression_bp: float = 0.0

    #: Share of the debt on which repression actually gains the Treasury
    #: anything: everything except the Federal Reserve's own holdings.
    #:
    #: Paying the Fed a below-market rate is exactly neutral, because the Fed
    #: remits its income back to the Treasury -- less interest in means smaller
    #: remittances out. CBO's net interest is already reported net of those
    #: remittances, so charging repression against the full stock would count
    #: a saving that does not exist. The Fed held $4.4tn of $32.4tn in
    #: September 2026, hence 0.86.
    repression_effective_share: float = 0.86

    #: One-off write-down of the debt stock in year 1, as a share of GDP.
    #: Used for default and for surprise-inflation revaluation.
    haircut_gdp: float = 0.0

    #: Share of the debt whose payments are indexed to inflation (TIPS were
    #: $2.15tn against $32.4tn held by the public in 2026).  Higher inflation
    #: raises the cost of this slice one-for-one, so it earns no debasement.
    #: Without it the model would credit inflation with eroding debt that is
    #: contractually protected from exactly that.
    indexed_share: float = 0.066

    #: The inflation rate the indexed slice is priced against -- the baseline.
    #: Only inflation *above* this raises its cost relative to the projection.
    indexed_baseline_inflation: float = 0.023

    #: Fiscal multiplier on the *impulse* -- the year-on-year change in the
    #: primary balance, not its level.  Tightening by one point of GDP in a
    #: year subtracts ``fiscal_multiplier`` points from that year's real
    #: growth.  Zero (the default) is the textbook identity, which flatters
    #: every consolidation; 0.6 is a reasonable central estimate for an
    #: advanced economy away from the zero lower bound, and 1.5 or more is the
    #: range estimated when policy rates cannot offset.
    fiscal_multiplier: float = 0.0

    #: Share of the output lost to a consolidation that is never recovered.
    #: At 0 the economy makes the level back the following year and only the
    #: timing changed; at 1 the loss is permanent and the debt ratio carries a
    #: smaller denominator forever.
    hysteresis: float = 0.5

    label: str = "scenario"


def _default_issuance() -> np.ndarray:
    """Approximate current Treasury issuance mix by maturity."""
    s = np.zeros(_MAX_MATURITY)
    s[0] = 0.62   # bills
    s[1] = 0.11   # 2y
    s[2] = 0.07   # 3y
    s[4] = 0.08   # 5y
    s[6] = 0.05   # 7y
    s[9] = 0.045  # 10y
    s[19] = 0.010  # 20y
    s[29] = 0.015  # 30y
    return s / s.sum()


#: Term premium by maturity: the extra yield a lender demands for locking up
#: money, over and above the average short rate expected over the same period.
#:
#: Taken from the slope of the Treasury curve of 4 September 2026, on the
#: reading that with the Fed near neutral the upward slope is compensation for
#: duration rather than an expectation of rising rates.
#:
#: The distinction matters for issuance scenarios.  If long rates were simply
#: "short plus a fixed spread", bills would be cheaper in every state of the
#: world and borrowing short would be a free lunch.  Long rates are instead
#: built below as the *expected average short rate over the life of the bond*
#: plus this premium, so a temporary spike in short rates inverts the curve --
#: which is exactly the state in which having funded short is expensive.
_TERM_PREMIUM = np.interp(
    np.arange(1, _MAX_MATURITY + 1),
    np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]),
    np.array([0.0, 0.15, 0.39, 0.47, 0.56, 0.67, 0.80, 1.27]) / 100.0,
)


@dataclass(frozen=True)
class Projection:
    """Year-by-year output of a scenario, all ratios as shares of GDP."""

    label: str
    years: np.ndarray
    debt_gdp: np.ndarray
    primary_gdp: np.ndarray
    interest_gdp: np.ndarray
    deficit_gdp: np.ndarray
    effective_rate: np.ndarray
    market_rate: np.ndarray
    nominal_gdp: np.ndarray
    average_maturity: np.ndarray

    @property
    def terminal_debt(self) -> float:
        return float(self.debt_gdp[-1])

    @property
    def peak_debt(self) -> float:
        return float(self.debt_gdp.max())

    def stabilises(self, tol: float = 0.002) -> bool:
        """True if the ratio is flat or falling over the final five years."""
        return bool(self.debt_gdp[-1] - self.debt_gdp[-6] <= tol * 5)


def project(
    ladder: DebtLadder,
    gdp0: float,
    assumptions: Assumptions,
    years: int = 20,
    start_year: int = 2026,
) -> Projection:
    """Roll the debt forward year by year under one set of assumptions.

    Each year: interest accrues on every outstanding vintage at its own
    coupon; the primary balance and that interest determine the deficit;
    maturing principal plus the deficit is issued anew across the maturity
    distribution at current market rates; and nominal GDP grows.
    """
    g = _as_path(assumptions.real_growth, years)
    pi = _as_path(assumptions.inflation, years)
    rr = _as_path(assumptions.real_short_rate, years)
    pb = _as_path(assumptions.primary_balance, years)
    shares = np.asarray(assumptions.issuance_shares, dtype=float)
    shares = shares / shares.sum()

    # Expected path of the short rate, extended past the horizon so that a
    # bond issued in the final year still has an expectation to price against.
    # Agents are assumed to foresee the scenario's own rate path, which is the
    # natural convention for a deterministic projection.
    err = (
        rr if assumptions.expected_real_short_rate is None
        else _as_path(assumptions.expected_real_short_rate, years)
    )
    epi = (
        pi if assumptions.expected_inflation is None
        else _as_path(assumptions.expected_inflation, years)
    )
    short_path = np.concatenate([err + epi, np.full(_MAX_MATURITY, err[-1] + epi[-1])])

    # Growth response to fiscal tightening.  The impulse is the change in the
    # primary balance; the demand hit lands in the same year and the part that
    # is not permanent is made back the year after.
    if assumptions.fiscal_multiplier:
        impulse = np.diff(np.concatenate([[pb[0]], pb]))
        hit = assumptions.fiscal_multiplier * impulse
        g = g - hit
        g[1:] = g[1:] + hit[:-1] * (1.0 - assumptions.hysteresis)

    principal = ladder.principal.copy()
    coupon = ladder.coupon.copy()
    gdp = float(gdp0)
    b0 = principal.sum() / gdp

    out = {k: np.zeros(years) for k in
           ("debt", "primary", "interest", "deficit", "eff", "mkt", "gdp", "wam")}

    for t in range(years):
        debt = principal.sum()

        # A one-off write-down lands at the start of the first year: an
        # outright default, or the revaluation from a surprise inflation.
        if t == 0 and assumptions.haircut_gdp:
            cut = min(assumptions.haircut_gdp * gdp, debt)
            principal *= (1.0 - cut / debt)
            debt = principal.sum()

        interest = float((principal * coupon).sum())
        # Indexed debt costs more when inflation runs above what it was
        # priced against; that slice cannot be inflated away.
        interest += (
            assumptions.indexed_share
            * debt
            * (pi[t] - assumptions.indexed_baseline_inflation)
        )
        primary = pb[t] * gdp
        deficit = interest - primary          # positive = borrowing need

        # What the market charges on new issuance this year.  A bond of
        # maturity m yields the average short rate expected over its life plus
        # the term premium; a bill yields the current short rate.
        adjust = (
            assumptions.risk_premium_per_10pp * max(0.0, debt / gdp - b0) / 0.10
            - (assumptions.repression_bp / 10_000.0)
            * assumptions.repression_effective_share
        )
        expected = np.array(
            [short_path[t:t + m].mean() for m in range(1, _MAX_MATURITY + 1)]
        )
        curve = expected + _TERM_PREMIUM + adjust
        # The one-year point is the realized short rate, not the expected one:
        # money borrowed today for a year costs today's rate.
        curve[0] = rr[t] + pi[t] + adjust
        short = curve[0]

        out["debt"][t] = debt / gdp
        out["primary"][t] = primary / gdp
        out["interest"][t] = interest / gdp
        out["deficit"][t] = deficit / gdp
        out["mkt"][t] = short
        out["gdp"][t] = gdp
        if debt > 0:
            out["eff"][t] = interest / debt
            out["wam"][t] = (
                np.arange(1, principal.size + 1) * principal
            ).sum() / debt
        else:  # the debt has been retired outright
            out["eff"][t] = short
            out["wam"][t] = 0.0

        # Roll the ladder down one year; the front bucket matures.
        maturing = principal[0]
        principal = np.concatenate([principal[1:], [0.0]])
        coupon = np.concatenate([coupon[1:], [0.0]])

        # Refinance what matured, plus this year's deficit.
        issuance = maturing + deficit
        if issuance > 0:
            add = issuance * shares
            total_new = principal + add
            with np.errstate(invalid="ignore", divide="ignore"):
                coupon = np.where(
                    total_new > 0,
                    (principal * coupon + add * curve) / np.maximum(total_new, 1e-12),
                    coupon,
                )
            principal = total_new
        else:
            # A primary surplus large enough to retire debt outright: pay down
            # the longest maturities first, which is what buybacks target.
            to_retire = -issuance
            for i in range(principal.size - 1, -1, -1):
                take = min(principal[i], to_retire)
                principal[i] -= take
                to_retire -= take
                if to_retire <= 0:
                    break

        gdp *= (1.0 + g[t]) * (1.0 + pi[t])

    return Projection(
        label=assumptions.label,
        years=np.arange(start_year, start_year + years),
        debt_gdp=out["debt"],
        primary_gdp=out["primary"],
        interest_gdp=out["interest"],
        deficit_gdp=out["deficit"],
        effective_rate=out["eff"],
        market_rate=out["mkt"],
        nominal_gdp=out["gdp"],
        average_maturity=out["wam"],
    )
