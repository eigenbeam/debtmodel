"""September 2026 starting point and CBO-anchored baseline for the projections.

The maturity ladder is the real one, derived from the same security-level
Treasury data as :mod:`debtmodel.calibration_2026`.  Everything else is set to
reproduce the Congressional Budget Office's February 2026 baseline, so that
"nothing changes" means something specific and externally checkable rather than
whatever the model happens to produce.
"""

from __future__ import annotations

import numpy as np

from .fiscal import Assumptions, DebtLadder, Projection, project

__all__ = [
    "GDP_2026", "DEBT_HELD_BY_PUBLIC", "CBO_TARGETS",
    "starting_ladder", "baseline", "run",
]

#: Nominal GDP, Q2 2026 SAAR, $bn.
GDP_2026 = 32_486.0

#: Debt held by the public at the start of the projection, $bn.
#: CBO puts it at 101% of GDP in fiscal 2026.
DEBT_HELD_BY_PUBLIC = 1.01 * GDP_2026

#: CBO February 2026 baseline, "The Budget and Economic Outlook: 2026 to 2036".
#: The validation target for the no-change case.
CBO_TARGETS = {
    2026: dict(debt=1.01, deficit=0.058, interest=0.033),
    2036: dict(debt=1.20, deficit=0.067, interest=0.046),
}

# Principal maturing by year and its average coupon, from the MSPD security
# detail of 31 August 2026 (see scripts/build_2026_profile.py).  Bills carry
# their yield as a coupon, since that is how they reach the budget.
_PRINCIPAL = np.array([
    9659.1, 3712.3, 2582.0, 1769.9, 2012.1, 1099.1,
    1040.3,  499.3,  552.6,  489.7,   38.0,   48.0,
     106.1,  307.0,  489.7,  383.2,  335.0,  340.6,
     345.6,  315.0,  156.1,  179.5,  225.6,  279.9,
     356.4,  323.5,  264.5,  288.3,  318.6,  267.8,
])
_COUPON = np.array([
    3.900, 2.793, 3.303, 3.264, 3.244, 3.153,
    3.916, 4.182, 4.344, 4.232, 4.891, 4.441,
    4.164, 2.956, 2.666, 2.798, 3.564, 3.991,
    3.873, 3.665, 2.902, 2.977, 2.846, 1.682,
    1.977, 2.459, 3.852, 4.465, 4.656, 4.829,
]) / 100.0

#: Interest cost divided by debt outstanding at the start: CBO's net interest
#: of 3.3% of GDP against debt of 101% of GDP.
INITIAL_EFFECTIVE_RATE = 0.033 / 1.01

#: CBO has the primary deficit narrowing slightly over the decade even as the
#: total deficit widens, because all of the widening is interest.  Fitted, with
#: the rate path below, to CBO's 2036 debt, deficit and interest figures
#: simultaneously.
_PRIMARY_PATH = np.linspace(-0.025, -0.023, 11)

#: Real short rate: 1.7% today, easing to 1.2% as the Fed reaches neutral.
#: This is what CBO's projected interest cost implies given the ladder -- their
#: 4.6%-of-GDP net interest in 2036 is an average rate of about 3.8%, well
#: below today's marginal rate, so their baseline has policy rates falling.
_REAL_SHORT_PATH = np.concatenate([np.linspace(0.017, 0.012, 5), np.full(25, 0.012)])


def starting_ladder() -> DebtLadder:
    """The real maturity ladder, scaled to debt held by the public."""
    raw = DebtLadder(_PRINCIPAL, _COUPON)
    return raw.rescaled(DEBT_HELD_BY_PUBLIC, INITIAL_EFFECTIVE_RATE)


def baseline(**overrides) -> Assumptions:
    """CBO-consistent no-change assumptions; keyword arguments override any field."""
    base = dict(
        real_growth=0.018,
        inflation=0.023,
        real_short_rate=_REAL_SHORT_PATH,
        primary_balance=_PRIMARY_PATH,
        label="baseline (CBO)",
    )
    base.update(overrides)
    return Assumptions(**base)


def run(assumptions: Assumptions, years: int = 20) -> Projection:
    """Project one scenario from the September 2026 starting point."""
    return project(starting_ladder(), GDP_2026, assumptions, years=years)
