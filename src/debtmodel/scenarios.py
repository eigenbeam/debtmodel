"""A spread of scenarios around the CBO-consistent baseline.

Grouped by what they do to the debt ratio rather than by how they work, since
that is the question worth asking: which levers actually bend the trajectory,
and how hard do they have to be pulled?

Every scenario changes the baseline in one identifiable way (or, in the last
group, a named combination), so the effects can be attributed.  Sizes are
chosen to be recognisable policy events rather than round numbers -- a
recession the size of 2008, a consolidation the size of the 1990s.
"""

from __future__ import annotations

import numpy as np

from .fiscal import Assumptions, Projection
from .fiscal_calibration import baseline, run

__all__ = ["SCENARIOS", "GROUPS", "run_all", "required_primary_balance"]

_YEARS = 25


def _recession_growth(start: int = 2, depth: float = 0.030,
                      years: int = _YEARS) -> np.ndarray:
    """Baseline growth with a two-year contraction and a partial catch-up."""
    g = np.full(years, 0.018)
    g[start] = 0.018 - depth
    g[start + 1] = 0.018 - depth / 2
    g[start + 2:start + 4] = 0.028          # recovery, never fully regained
    return g


def _recession_primary(start: int = 2, years: int = _YEARS) -> np.ndarray:
    """Automatic stabilisers plus discretionary response, then a slow unwind."""
    pb = np.linspace(-0.025, -0.023, 11)
    pb = np.concatenate([pb, np.full(years - 11, -0.023)])
    pb[start:start + 2] -= 0.030
    pb[start + 2:start + 5] -= np.array([0.020, 0.012, 0.006])
    return pb


def _phased(start: float, end: float, over: int, years: int = _YEARS) -> np.ndarray:
    """Move from ``start`` to ``end`` over ``over`` years, then hold."""
    return np.concatenate([np.linspace(start, end, over), np.full(years - over, end)])


#: Issuance concentrated at the very front of the curve.
_BILLS = np.eye(30)[0] * 0.85 + np.eye(30)[1] * 0.15

#: A Volcker-style episode: the Fed drives the real short rate up 350bp for
#: three years, then eases back to a level above where it started.  This
#: inverts the curve, since the term spread over the short rate is fixed.
_SPIKE = np.concatenate([
    [0.017, 0.030, 0.052, 0.052, 0.045, 0.035],
    np.full(19, 0.022),
])

#: The spike is a surprise: bonds are priced off the baseline path, so debt
#: issued before it locked in cheap coupons.  This is what duration buys.
_EXPECTED = np.concatenate([np.linspace(0.017, 0.012, 5), np.full(20, 0.012)])


#: All scenarios, keyed by short name.
SCENARIOS: dict[str, Assumptions] = {
    # -- nothing changes ----------------------------------------------------
    "baseline": baseline(label="Baseline (CBO)"),

    # -- makes it worse -----------------------------------------------------
    "rates_up": baseline(
        real_short_rate=_phased(0.017, 0.022, 5),
        label="Term premium +100bp",
    ),
    "recession": baseline(
        real_growth=_recession_growth(),
        primary_balance=_recession_primary(),
        label="Recession in 2028",
    ),
    "stagnation": baseline(
        real_growth=0.010,
        label="Growth disappoints (1.0%)",
    ),
    "expansion": baseline(
        primary_balance=_phased(-0.025, -0.038, 3),
        label="Fiscal expansion (+1.5% GDP)",
    ),
    "spiral": baseline(
        risk_premium_per_10pp=0.0025,
        label="Investors price the debt (25bp/10pp)",
    ),

    # -- roughly neutral ----------------------------------------------------
    "twist": baseline(
        issuance_shares=_BILLS,
        label="Treasury Twist, rates as forecast",
    ),
    # The Twist's real exposure is not a parallel shift -- bills stay cheaper
    # than bonds under one -- but a Fed forced to spike short rates to fight
    # inflation, which inverts the curve and makes short funding the expensive
    # kind.  Both legs below share the shock; only the issuance mix differs.
    "spike_long_funded": baseline(
        real_short_rate=_SPIKE,
        expected_real_short_rate=_EXPECTED,
        label="Rate spike, current issuance mix",
    ),
    "spike_bill_funded": baseline(
        issuance_shares=_BILLS,
        real_short_rate=_SPIKE,
        expected_real_short_rate=_EXPECTED,
        label="Rate spike, after the Twist",
    ),
    "inflation_expected": baseline(
        inflation=_phased(0.023, 0.043, 3),
        label="Inflation +2pp, anticipated",
    ),
    "inflation_surprise": baseline(
        haircut_gdp=0.026,
        label="Inflation +1pp, surprise (one-off)",
    ),

    # -- stabilises ---------------------------------------------------------
    # These two move the primary balance, so they carry a fiscal multiplier;
    # the rest vary conditions rather than policy, where an impulse-based
    # multiplier has nothing to act on.
    "consolidation": baseline(
        primary_balance=_phased(-0.025, 0.003, 6),
        fiscal_multiplier=0.6,
        label="Primary balance to +0.3% (2.8pp)",
    ),
    "growth_boom": baseline(
        real_growth=_phased(0.018, 0.031, 4),
        label="Productivity boom (3.1% growth)",
    ),
    "repression": baseline(
        repression_bp=175.0,
        label="Financial repression (-175bp)",
    ),

    # -- reduces ------------------------------------------------------------
    "postwar": baseline(
        inflation=_phased(0.023, 0.040, 3),
        repression_bp=250.0,
        primary_balance=_phased(-0.025, -0.005, 6),
        fiscal_multiplier=0.6,
        label="Postwar playbook",
    ),
    "grand_bargain": baseline(
        primary_balance=_phased(-0.025, 0.020, 8),
        real_growth=_phased(0.018, 0.022, 5),
        fiscal_multiplier=0.6,
        label="Grand bargain (4.5pp swing)",
    ),
    "default": baseline(
        haircut_gdp=0.30,
        risk_premium_per_10pp=0.0,
        real_short_rate=_phased(0.017, 0.042, 2),
        real_growth=_recession_growth(start=0, depth=0.050),
        label="Outright default (30% of GDP)",
    ),
}

#: Which group each scenario is intended to illustrate.
GROUPS: dict[str, list[str]] = {
    "Baseline": ["baseline"],
    "Makes it worse": ["rates_up", "recession", "stagnation", "expansion", "spiral"],
    "Roughly neutral": ["twist", "inflation_expected", "inflation_surprise"],
    "Issuance mix under stress": ["spike_long_funded", "spike_bill_funded"],
    "Stabilises": ["consolidation", "growth_boom", "repression"],
    "Reduces": ["postwar", "grand_bargain", "default"],
}


def run_all(years: int = _YEARS) -> dict[str, Projection]:
    """Project every scenario."""
    return {k: run(v, years=years) for k, v in SCENARIOS.items()}


def required_primary_balance(
    years: int = _YEARS, tol: float = 1e-5, **overrides
) -> float:
    """The constant primary balance that leaves debt/GDP where it started.

    Solved by bisection.  Answers "what would it take?" for any set of
    background assumptions -- the number that matters most for judging whether
    a consolidation is politically conceivable.
    """
    lo, hi = -0.10, 0.15
    for _ in range(60):
        mid = (lo + hi) / 2
        p = run(baseline(primary_balance=mid, **overrides), years=years)
        if p.debt_gdp[-1] > p.debt_gdp[0]:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2
