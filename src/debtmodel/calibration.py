"""US 2012 calibration, after Hilscher, Raviv & Reis (2014).

Everything here is a number taken from, or constructed to match, the paper's
end-of-2012 snapshot.  Provenance is noted per item, because the three inputs
have very different epistemic status: the nominal yield curve is public data,
the inflation term structures are the paper's published table 4, and the
maturity profile is a *stylized reconstruction* -- the underlying CRSP
security-level data is not in the paper, so the profile here is built to
reproduce the paper's published moments and should be replaced with real data
before any result is leaned on.
"""

from __future__ import annotations

import numpy as np

from .burden import DebasementModel
from .curves import TermStructure
from .maturity import DebtProfile

__all__ = [
    "GDP_2012",
    "NOMINAL_CURVE_2012",
    "INFLATION_BASELINE",
    "INFLATION_COUNTERFACTUALS",
    "COUNTERFACTUAL_LABELS",
    "PAPER_TABLE_3_PRIVATE",
    "private_holdings",
    "baseline_model",
    "counterfactual_model",
]

#: US nominal GDP in 2012, $bn.  Denominator for every "% of GDP" figure.
GDP_2012 = 16_200.0

#: Treasury constant-maturity nominal yields, 31 December 2012 (decimal).
#: Public data.  This is what actually prices the debt, so the baseline
#: burden reproduces the observed market value by construction.
NOMINAL_CURVE_2012 = TermStructure(
    tenors=np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]),
    rates=np.array([0.05, 0.11, 0.14, 0.25, 0.36, 0.72, 1.18, 1.78, 2.54, 2.95])
    / 100.0,
    name="nominal 2012",
)

#: Horizons at which the paper's table 4 reports inflation.
_INFLATION_TENORS = np.array([1.0, 3.0, 5.0, 30.0])

#: Risk-adjusted harmonic mean of average annual inflation, table 4 baseline row.
INFLATION_BASELINE = TermStructure(
    _INFLATION_TENORS,
    np.array([1.5, 2.1, 2.4, 3.0]) / 100.0,
    name="baseline inflation",
)

#: The seven counterfactual inflation term structures of table 4.
INFLATION_COUNTERFACTUALS: dict[int, TermStructure] = {
    1: TermStructure(_INFLATION_TENORS, np.array([3.2, 4.0, 4.5, 5.6]) / 100.0,
                     name="permanently higher"),
    2: TermStructure(_INFLATION_TENORS, np.array([4.4, 4.7, 5.2, 5.2]) / 100.0,
                     name="right tail only"),
    3: TermStructure(_INFLATION_TENORS, np.array([3.1, 4.0, 4.5, 5.6]) / 100.0,
                     name="higher and more variable"),
    4: TermStructure(_INFLATION_TENORS, np.array([3.2, 4.0, 4.6, 5.7]) / 100.0,
                     name="higher for sure"),
    5: TermStructure(_INFLATION_TENORS, np.array([3.0, 2.9, 3.2, 3.3]) / 100.0,
                     name="partially anticipated"),
    6: TermStructure(_INFLATION_TENORS, np.array([3.2, 3.3, 3.2, 3.1]) / 100.0,
                     name="temporary increase"),
    7: TermStructure(_INFLATION_TENORS, np.array([1.5, 2.5, 3.4, 5.5]) / 100.0,
                     name="gradual increase"),
}

COUNTERFACTUAL_LABELS = {k: v.name for k, v in INFLATION_COUNTERFACTUALS.items()}

#: Table 3, "privately held" column: fall in the real PV of debt, share of GDP.
#: The target this model is checked against, not an input to it.
PAPER_TABLE_3_PRIVATE = {
    1: 0.037, 2: 0.044, 3: 0.034, 4: 0.038, 5: 0.013, 6: 0.012, 7: 0.023,
}

# Stylized annual payment profile, shape read off figure 2 (privately-held
# line), then rescaled so the model reproduces the paper's moments.  The
# shape is the part that matters and the part that is least certain.
_PROFILE_SHAPE = np.array(
    [2800, 1000, 950, 700, 730, 400, 350, 250, 200, 200,
     50, 45, 45, 45, 45, 45, 40, 40, 40, 40,
     40, 40, 40, 45, 50, 100, 110, 120, 130, 150],
    dtype=float,
)

# Figure 2's long end is the hardest region to read and the profile as drawn
# overstates it, giving too long an average maturity.  Damped to match the
# paper's reported 4.8-year average.
_LONG_END_DAMPING = 0.55

# Payments are bucketed by year but arrive throughout it, so each bucket sits
# at its midpoint.  This is not cosmetic: putting the first bucket at t=1
# instead of t=0.5 doubles its (small) contribution to debasement, and the
# first bucket is by far the largest.
_HORIZONS = np.arange(1.0, 31.0) - 0.5


def private_holdings() -> DebtProfile:
    """Nominal payments due to private holders of US Treasuries, end-2012, $bn.

    Calibrated to reproduce three published figures: a market value of $8.3tn
    (51.2% of GDP), an average maturity of 4.8 years, and a 3.7%-of-GDP
    debasement under counterfactual 1.
    """
    payments = _PROFILE_SHAPE.copy()
    payments[_HORIZONS > 25.0] *= _LONG_END_DAMPING
    profile = DebtProfile(payments, _HORIZONS, label="privately held")

    # Scale so the market value matches the paper's 51.2% of GDP.
    target = 0.512 * GDP_2012
    pricer = DebasementModel.from_nominal_curve(NOMINAL_CURVE_2012, INFLATION_BASELINE)
    return profile.scaled(target / pricer.value(profile))


def baseline_model() -> DebasementModel:
    """The end-2012 model: observed nominal curve, table 4 baseline inflation."""
    return DebasementModel.from_nominal_curve(NOMINAL_CURVE_2012, INFLATION_BASELINE)


def counterfactual_model(n: int) -> DebasementModel:
    """Baseline real rates with the inflation term structure of counterfactual ``n``.

    Real rates are held at their baseline values, so the change in the debt
    burden is attributable to inflation alone.
    """
    if n not in INFLATION_COUNTERFACTUALS:
        raise KeyError(
            f"counterfactual {n} not in {sorted(INFLATION_COUNTERFACTUALS)}"
        )
    return baseline_model().with_inflation(INFLATION_COUNTERFACTUALS[n])
