"""Scenarios for the one lever that is a direct policy choice.

Growth, interest rates and inflation are conditions; the primary balance is a
decision. This module varies it along the four dimensions that matter for that
decision -- how large, how fast, how soon, and how durable -- and prices the
growth cost of getting there.

Sizes are quoted as the *swing* from today's primary deficit of 2.5% of GDP.
A "2.8-point consolidation" therefore ends at a primary surplus of 0.3%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fiscal import Assumptions, Projection
from .fiscal_calibration import baseline, run

__all__ = [
    "START_BALANCE", "MULTIPLIER", "HYSTERESIS",
    "consolidation", "path", "required_for_target", "cost_of_delay",
    "SIZES", "SPEEDS", "DELAYS", "TARGETS",
    "BUDGET_2026", "BUDGET_2036", "budget_equivalents", "entitlement_drift",
    "HISTORICAL", "package", "PACKAGES", "repression_transfer",
    "repression_frontier", "domestic_burden", "EFFECTIVE_SHARE",
    "JAPAN", "US_HOLDERS",
]

#: Today's primary balance, share of GDP (CBO, fiscal 2026).
START_BALANCE = -0.025

#: Central fiscal multiplier. Away from the zero lower bound -- policy rates
#: are near 4%, so the central bank has room to offset -- the empirical range
#: clusters around 0.5-0.8 for advanced economies.
MULTIPLIER = 0.6

#: Share of the output loss never recovered.
HYSTERESIS = 0.5

_YEARS = 25


def path(size: float, over: int = 6, start: int = 1,
         lapses_after: int | None = None, years: int = _YEARS) -> np.ndarray:
    """A primary-balance path: tighten by ``size`` of GDP over ``over`` years.

    Parameters
    ----------
    size:
        Total swing as a share of GDP, e.g. ``0.028`` for 2.8 points.
    over:
        Years taken to phase it in.
    start:
        Years from now before the first tightening lands.  Delay is the most
        expensive parameter here, because the debt compounds while nothing
        happens and the required swing grows.
    lapses_after:
        If given, the consolidation unwinds linearly over the five years
        beginning this many years from now -- a consolidation that does not
        stick.
    """
    p = np.full(years, START_BALANCE)
    end = START_BALANCE + size
    for t in range(years):
        if t < start:
            p[t] = START_BALANCE
        elif t < start + over:
            p[t] = START_BALANCE + size * (t - start + 1) / over
        else:
            p[t] = end
    if lapses_after is not None:
        for t in range(lapses_after, years):
            frac = min(1.0, (t - lapses_after + 1) / 5.0)
            p[t] = end + (START_BALANCE - end) * frac
    return p


def consolidation(size: float, over: int = 6, start: int = 1,
                  lapses_after: int | None = None,
                  multiplier: float = MULTIPLIER, label: str | None = None,
                  **overrides) -> Assumptions:
    """Assumptions for a consolidation of a given size, speed and timing."""
    return baseline(
        primary_balance=path(size, over, start, lapses_after),
        fiscal_multiplier=multiplier,
        hysteresis=HYSTERESIS,
        label=label or f"{size:+.1%} over {over}y",
        **overrides,
    )


def required_for_target(target: float, years: int = _YEARS,
                        over: int = 6, start: int = 1,
                        multiplier: float = MULTIPLIER, **overrides) -> float:
    """The consolidation *size* that lands debt/GDP on ``target`` at the horizon.

    Returns the swing from today's primary balance, as a share of GDP.
    """
    lo, hi = -0.03, 0.15
    for _ in range(60):
        mid = (lo + hi) / 2
        p = run(consolidation(mid, over, start, multiplier=multiplier, **overrides),
                years=years)
        if p.debt_gdp[-1] > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2


def cost_of_delay(target: float = 1.01, delays=(1, 5, 10, 15),
                  years: int = _YEARS) -> list[tuple[int, float]]:
    """How much larger the swing must be for each year of waiting.

    Held to the same 2050 destination, so the comparison is like for like:
    the later you start, the more you must do to arrive at the same place.
    """
    return [(d, required_for_target(target, years=years, start=d)) for d in delays]


#: How large -- swings from today's 2.5%-of-GDP primary deficit.
SIZES = [0.010, 0.020, 0.028, 0.040, 0.050]

#: How fast -- years to phase in a 2.8-point swing.
SPEEDS = [2, 6, 10, 15]

#: How soon -- years before a 2.8-point swing begins.
DELAYS = [1, 5, 10, 15]

#: Where to land debt/GDP by 2050.
TARGETS = {
    "hold at today's 101%": 1.01,
    "back to 90%": 0.90,
    "back to the 60% norm": 0.60,
}


# ---------------------------------------------------------------------------
# What a swing means in budget terms
# ---------------------------------------------------------------------------

#: CBO February 2026 baseline, fiscal 2026, shares of GDP.
BUDGET_2026 = {
    "revenues": 0.175,          # $5.6tn
    "outlays": 0.233,           # $7.4tn
    "net interest": 0.033,      # $1.0tn
    "Social Security": 0.052,
    "Medicare": 0.040,
}

#: Projected 2036 shares, for the drift calculation below.
BUDGET_2036 = {
    "revenues": 0.178,
    "outlays": 0.244,
    "net interest": 0.046,
    "Social Security": 0.059,
    "Medicare": 0.052,
}

GDP_2026 = 32_486.0             # $bn
HOUSEHOLDS = 132.0e6            # approximate, US 2026


def budget_equivalents(size: float) -> dict[str, float]:
    """Translate a swing of ``size`` share of GDP into budget quantities."""
    dollars = size * GDP_2026
    non_interest = BUDGET_2026["outlays"] - BUDGET_2026["net interest"]
    return {
        "$bn per year": dollars,
        "share of revenues": size / BUDGET_2026["revenues"],
        "share of outlays": size / BUDGET_2026["outlays"],
        "share of non-interest outlays": size / non_interest,
        "$ per household per year": dollars * 1e9 / HOUSEHOLDS,
    }


def entitlement_drift() -> float:
    """Rise in Social Security plus Medicare as a share of GDP, 2026 to 2036.

    This is the part of the problem that arrives without any decision being
    taken, and it is why the gross policy effort exceeds the net improvement:
    holding the primary balance merely flat already requires offsetting it.
    """
    return (
        (BUDGET_2036["Social Security"] - BUDGET_2026["Social Security"])
        + (BUDGET_2036["Medicare"] - BUDGET_2026["Medicare"])
    )


#: Comparable episodes, as reported in the IMF and GAO literature.  These are
#: improvements in the *overall* balance except where noted, so they are not
#: exactly like-for-like with a primary-balance swing -- in each case falling
#: interest costs did some of the work, which flatters them relative to the US
#: today, where interest is rising.
HISTORICAL = [
    dict(label="United States, 1993-98", swing=0.050, years=6,
         note="OBRA-90 and OBRA-93; overall balance improved 5 points of GDP, "
              "reaching surplus by 1998. Roughly half tax, half spending, "
              "with a strong expansion and falling rates helping throughout."),
    dict(label="Canada, 1995-97", swing=0.050, years=3,
         note="Chiefly expenditure-based, from deficits above 9% of GDP and "
              "debt above 100%. Programme spending fell about 11 points of GDP "
              "over the decade; surplus by 1998, sustained to 2008."),
    dict(label="Sweden, 1994-2000", swing=0.092, years=7,
         note="Cyclically adjusted primary balance improved 9.2 points -- "
              "spending cuts near 16 points of GDP against 0.8 of revenue. "
              "Followed a currency crisis and depreciation; growth averaged "
              "6.3% over the period."),
]


# ---------------------------------------------------------------------------
# Consolidation combined with financial repression
# ---------------------------------------------------------------------------

#: Debt held by the public, $bn, September 2026.
DEBT_HELD_BY_PUBLIC = 32_811.0

#: Federal Reserve holdings, $bn (SOMA, 2 September 2026).
FED_HOLDINGS = 4_446.0

#: Foreign holdings as a share of publicly held debt (TIC, December 2025).
FOREIGN_SHARE = 0.31

#: Share of the debt on which repression gains the Treasury anything.
EFFECTIVE_SHARE = 1.0 - FED_HOLDINGS / DEBT_HELD_BY_PUBLIC


def repression_transfer(bp: float) -> dict[str, float]:
    """The annual transfer that ``bp`` of repression takes from bondholders.

    Repression is a tax. It does not create resources, it moves them: the
    Treasury pays a below-market rate and whoever holds the debt receives one.
    Sizing that transfer is the only way to compare repression honestly
    against a consolidation of the same effect, which is an explicit tax or
    spending cut of a similar magnitude.

    The Federal Reserve's share is excluded -- repressing it is neutral -- and
    the remainder is split between foreign and domestic holders.
    """
    base = DEBT_HELD_BY_PUBLIC - FED_HOLDINGS
    total = bp / 10_000.0 * base
    foreign_of_base = FOREIGN_SHARE * DEBT_HELD_BY_PUBLIC / base
    return {
        "$bn per year": total,
        "% of GDP": total / GDP_2026,
        "from foreign holders, $bn": total * foreign_of_base,
        "from domestic holders, $bn": total * (1.0 - foreign_of_base),
        "exported share": foreign_of_base,
    }


def package(size: float, repression_bp: float, over: int = 6, start: int = 1,
            label: str | None = None, **overrides) -> Assumptions:
    """A consolidation of ``size`` combined with ``repression_bp`` of repression."""
    return consolidation(
        size, over=over, start=start, repression_bp=repression_bp,
        label=label or f"{size:+.1%} + {repression_bp:.0f}bp",
        **overrides,
    )


def domestic_burden(size: float, repression_bp: float) -> dict[str, float]:
    """Total annual cost of a package, split by who actually pays it.

    The explicit leg -- taxes raised or spending cut -- falls on residents in
    full. The implicit leg falls on bondholders, roughly a third of whom are
    foreign, so part of a repression-heavy package is exported.
    """
    explicit = size * GDP_2026
    t = repression_transfer(repression_bp)
    return {
        "explicit, $bn": explicit,
        "implicit, $bn": t["$bn per year"],
        "total, $bn": explicit + t["$bn per year"],
        "borne domestically, $bn": explicit + t["from domestic holders, $bn"],
        "borne abroad, $bn": t["from foreign holders, $bn"],
    }


def repression_frontier(levels=(0, 50, 100, 150, 200, 250, 300),
                        target: float = 1.01) -> list[dict]:
    """Consolidation required to hold debt flat, at each level of repression.

    The substitution is close to linear and close to one-for-one: with debt
    near 100% of GDP, a basis point off the interest rate is worth about a
    basis point of GDP in primary balance. What changes between the two is not
    the size of the adjustment but who bears it.
    """
    out = []
    for bp in levels:
        size = required_for_target(target, repression_bp=bp)
        out.append(dict(bp=bp, size=size, **domestic_burden(max(size, 0.0), bp)))
    return out


#: Five packages that all land on the same place -- debt back at today's 101%
#: of GDP in 2050 -- differing only in how the work is split between an
#: explicit consolidation and an implicit levy on bondholders.
#:
#: Because they share a destination, the comparison is clean: what changes
#: across the list is not how much adjustment happens but who pays for it.
PACKAGES = [
    ("All consolidation", 0.0254, 0),
    ("Two thirds consolidation", 0.0169, 100),
    ("Half and half", 0.0105, 175),
    ("Mostly repression", 0.0042, 250),
    ("All repression", 0.0, 300),
]


#: Japan, the counter-example. Repression carried about as far as a rich
#: democracy has taken it, with the debt ratio still twice the US level.
#:
#: The instructive number is the third one: Japan services twice America's
#: debt ratio for less than a third of America's interest bill. Repression
#: makes a very large debt cheap to carry. What it did not do, for three
#: decades, was stop the ratio rising -- that only happened once the primary
#: balance reached zero, which Japan reached in FY2025 and the US has not.
JAPAN = {
    "gross debt, % of GDP": 2.044,
    "net debt, % of GDP": 1.343,
    "interest, % of GDP": 0.010,
    "implied average rate": 0.010 / 2.044,
    "Bank of Japan share of JGBs and T-bills": 0.422,
    "Bank of Japan share of JGBs alone": 0.479,
    "banks": 0.168,
    "insurers": 0.137,
    "public and private pensions": 0.092,
    "foreign": 0.137,
    "households": 0.017,
}

#: The same holder split for the United States, for comparison.
US_HOLDERS = {
    "Federal Reserve share": FED_HOLDINGS / DEBT_HELD_BY_PUBLIC,
    "foreign": FOREIGN_SHARE,
}
