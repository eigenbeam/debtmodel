"""US September 2026 calibration.

A second parameter set for the same model.  Nothing in
:mod:`debtmodel.calibration` (the 2012 replication of Hilscher, Raviv & Reis)
is touched or imported for its values -- the two live side by side so the
structural change in the US fiscal position between them can be read off
directly.

Provenance is stronger here than for 2012 in one important respect and weaker
in another.

Stronger: the maturity profile is not reconstructed.  It is built from the
Treasury's Monthly Statement of the Public Debt at security level -- every
bill, note and bond outstanding, expanded into its actual coupon and principal
payments -- netted CUSIP by CUSIP against the New York Fed's SOMA holdings.
The 2012 profile had to be read off a chart.  See ``scripts/build_2026_profile.py``.

Also stronger: real rates are observed.  In 2012 the TIPS curve had to be
inferred by deflating the nominal curve with the paper's option-implied
inflation.  Here both the nominal and the real Treasury curves are published,
so inflation compensation is their ratio rather than an assumption.

Weaker: there are no option-implied inflation densities.  The paper recovered
risk-adjusted distributions from inflation caps and floors (its sections 4-5);
this module has only breakeven inflation, which is a risk-adjusted *mean*.  For
a fixed-parameter model that is the object needed anyway, but it means the
counterfactuals here cannot be derived from 2026 market data the way the
paper's were.  They are transplanted -- see :data:`HRR_CF1_SHIFT`.
"""

from __future__ import annotations

import numpy as np

from .burden import DebasementModel
from .curves import TermStructure
from .maturity import DebtProfile

__all__ = [
    "AS_OF",
    "GDP_2026",
    "NOMINAL_CURVE_2026",
    "REAL_CURVE_2026",
    "INFLATION_BREAKEVEN_2026",
    "HRR_CF1_SHIFT",
    "DEBT_STOCK_2026",
    "private_holdings_2026",
    "baseline_model_2026",
    "counterfactual_cf1_2026",
]

#: Valuation date.  Debt is as of the 31 Aug 2026 MSPD, curves as of 4 Sep 2026,
#: SOMA holdings as of 2 Sep 2026 -- the latest published on each.
AS_OF = "2026-09-08"

#: US nominal GDP, Q2 2026 seasonally-adjusted annual rate, $bn (BEA second
#: estimate, released 26 Aug 2026).
GDP_2026 = 32_486.0

#: Headline debt stock figures from the MSPD, 31 August 2026, $bn.
#: Recorded for reporting; the model uses the payment profile below.
DEBT_STOCK_2026 = {
    "total public debt outstanding": 40_175.6,
    "debt held by the public": 32_415.0,
    "intragovernmental holdings": 7_760.7,
    "marketable, held by public": 31_807.1,
    "  of which bills": 7_247.9,
    "  of which notes": 16_214.7,
    "  of which bonds": 5_512.7,
    "  of which TIPS": 2_152.0,
    "  of which floating rate notes": 679.9,
    "Federal Reserve (SOMA) Treasury holdings": 4_446.0,
}

#: Treasury constant-maturity nominal par yields, 4 September 2026 (decimal).
NOMINAL_CURVE_2026 = TermStructure(
    tenors=np.array([0.0833, 0.1667, 0.25, 0.3333, 0.5, 1.0, 2.0, 3.0,
                     5.0, 7.0, 10.0, 20.0, 30.0]),
    rates=np.array([3.79, 3.83, 3.90, 3.91, 4.00, 3.98, 4.13, 4.37,
                    4.45, 4.54, 4.65, 4.78, 5.25]) / 100.0,
    name="nominal 2026",
)

#: Treasury constant-maturity *real* (TIPS) par yields, 4 September 2026.
#: Published only from 5 years out; shorter horizons are filled by assuming
#: inflation compensation is flat back to the 1-year breakeven, which market
#: quotes put at 2.31% on the same date.
REAL_CURVE_2026 = TermStructure(
    tenors=np.array([5.0, 7.0, 10.0, 20.0, 30.0]),
    rates=np.array([2.17, 2.29, 2.43, 2.77, 2.96]) / 100.0,
    name="real 2026 (TIPS)",
)

#: Breakeven inflation, ``(1 + nominal) / (1 + real) - 1`` where both curves are
#: published, pinned at the quoted 2.31% one-year breakeven at the short end.
#:
#: The dip at 20 years is not a data error: the 20-year nominal point sits well
#: below the 30-year (4.78 against 5.25), a long-standing kink in that sector,
#: and it carries through to the breakeven.
INFLATION_BREAKEVEN_2026 = TermStructure(
    tenors=np.array([1.0, 5.0, 7.0, 10.0, 20.0, 30.0]),
    rates=np.array([2.31, 2.23, 2.20, 2.17, 1.96, 2.22]) / 100.0,
    name="breakeven inflation 2026",
)

#: The paper's counterfactual 1 expressed as an *increment*, from table 4:
#: the "permanently higher inflation" experiment raised risk-adjusted average
#: inflation by 1.7pp at one year rising to 2.6pp at thirty.
#:
#: Transplanting it onto the 2026 baseline asks "what would the paper's own
#: experiment do to today's debt?", which is the comparison worth making.  It
#: is emphatically not an estimate of what 2026 markets think possible: the
#: paper pinned the size of its shift to the 90th percentile of an
#: option-implied density that is not reconstructed here.
HRR_CF1_SHIFT = TermStructure(
    tenors=np.array([1.0, 3.0, 5.0, 30.0]),
    rates=np.array([1.7, 1.9, 2.1, 2.6]) / 100.0,
    name="HRR counterfactual 1 shift",
)

# Nominal payments due to holders outside the Federal Reserve, $bn per year,
# generated by scripts/build_2026_profile.py from the MSPD and SOMA.
#
# Principal and coupon on all marketable bills, notes, bonds and FRNs, less
# the Fed's holdings of each CUSIP.  TIPS ($2.15tn) are excluded: their
# payments are indexed, so inflation does not erode them -- equation (5) of the
# paper shows the debasement question turns on nominal debt alone.  FRNs are
# charged to their next quarterly reset rather than their maturity, since a
# floating coupon is not a payment fixed in nominal terms beyond one period.
_PAYMENTS_2026 = np.array([
    9622.2, 3459.1, 2674.8, 1889.7, 2069.9, 1189.3,
    1148.1,  656.7,  643.3,  553.3,  142.0,  144.5,
     161.3,  264.0,  402.7,  355.7,  359.9,  363.7,
     348.8,  313.8,  156.0,  216.6,  240.6,  275.3,
     339.4,  298.0,  276.5,  306.2,  296.8,  239.9,
], dtype=float)

#: Annual buckets sit at their midpoints, as in the 2012 calibration.
_HORIZONS = np.arange(1.0, 31.0) - 0.5


def private_holdings_2026() -> DebtProfile:
    """Nominal payments due to holders outside the Federal Reserve, $bn.

    Note this is a slightly wider set than the paper's "privately held": it
    still includes debt held by state and local governments, which the paper
    further removed (it put them at about 7.5% of privately-held debt in 2012).
    There is no comparably clean current source for that split, so it is left
    in and flagged rather than guessed at.  Scale the profile down if you want
    the narrower measure.
    """
    return DebtProfile(_PAYMENTS_2026, _HORIZONS, label="held outside the Fed, 2026")


def baseline_model_2026() -> DebasementModel:
    """The September 2026 model: observed nominal curve, breakeven inflation.

    Built from the nominal curve so the baseline burden reproduces the market
    value of the debt; the implied real rates then track the published TIPS
    curve, which is checked in the tests rather than assumed.
    """
    return DebasementModel.from_nominal_curve(
        NOMINAL_CURVE_2026, INFLATION_BREAKEVEN_2026
    )


def counterfactual_cf1_2026() -> DebasementModel:
    """2026 baseline with the paper's counterfactual-1 inflation shift applied."""
    return baseline_model_2026().shift_inflation(HRR_CF1_SHIFT)


#: Par value of nominal marketable debt outstanding net of Fed holdings, $bn,
#: computed from the same security list that generated ``_PAYMENTS_2026``.
#: The payment profile exceeds this by the coupons still to be paid.
PAR_NOMINAL_NET_FED = 24_660.0

#: Par-weighted average maturity of *all* marketable debt implied by the parsed
#: security list, in months.  Treasury publishes this statistic independently
#: and put it near 70 months at end-2025, so it is a genuine check that the
#: security-level data was read correctly.
IMPLIED_WAM_MONTHS = 71.3
