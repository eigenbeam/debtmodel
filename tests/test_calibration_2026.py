"""Checks on the September 2026 parameter set.

Two kinds of assertion appear below and they are not equally informative.
Some are genuine external checks -- the implied weighted average maturity
against Treasury's separately published figure, the payment profile against
the MSPD stock totals.  Others are round-trips that hold by construction and
are here to catch regressions, not to validate anything; those say so.
"""

import numpy as np
import pytest

from debtmodel.calibration_2026 import (
    DEBT_STOCK_2026,
    GDP_2026,
    HRR_CF1_SHIFT,
    IMPLIED_WAM_MONTHS,
    INFLATION_BREAKEVEN_2026,
    NOMINAL_CURVE_2026,
    PAR_NOMINAL_NET_FED,
    REAL_CURVE_2026,
    baseline_model_2026,
    counterfactual_cf1_2026,
    private_holdings_2026,
)


@pytest.fixture(scope="module")
def debt():
    return private_holdings_2026()


@pytest.fixture(scope="module")
def model():
    return baseline_model_2026()


# -- the debt data, against independently published figures -------------------


def test_implied_wam_matches_treasury_published_figure():
    """Treasury put the WAM of marketable debt near 70 months at end-2025.

    This is the real check on the security-level parse: the number falls out
    of 350-odd individual bills, notes and bonds and is not targeted anywhere.
    """
    assert IMPLIED_WAM_MONTHS == pytest.approx(70.0, abs=2.5)


def test_stock_figures_are_internally_consistent():
    """The MSPD summary table must add up."""
    s = DEBT_STOCK_2026
    assert s["debt held by the public"] + s["intragovernmental holdings"] == (
        pytest.approx(s["total public debt outstanding"], abs=1.0)
    )
    components = sum(
        s[k] for k in s if k.startswith("  of which")
    )
    assert components == pytest.approx(s["marketable, held by public"], rel=0.001)


def test_payment_profile_exceeds_par_by_coupons(debt):
    """Payments are principal plus coupon, so they must exceed par."""
    assert debt.face_value > PAR_NOMINAL_NET_FED
    coupons = debt.face_value - PAR_NOMINAL_NET_FED
    # Roughly 5 years of average maturity at a low-to-mid single digit coupon.
    assert 2_000.0 < coupons < 8_000.0


def test_debt_trades_below_par(model, debt):
    """Much of the outstanding stock was issued in 2020-21 at very low coupons.

    With the curve now at 4-5%, the aggregate must price at a discount.
    """
    assert model.value(debt) < PAR_NOMINAL_NET_FED
    assert model.value(debt) / PAR_NOMINAL_NET_FED > 0.85


def test_burden_as_share_of_gdp(model, debt):
    assert model.value(debt) / GDP_2026 == pytest.approx(0.728, abs=0.01)


# -- the curves ---------------------------------------------------------------


def test_real_rates_are_positive_and_upward_sloping(model):
    """The structural break from 2012, when the real curve was negative to 20y."""
    r = model.real_rates.at(np.array([5.0, 10.0, 30.0]))
    assert np.all(r > 0.015)
    assert np.all(np.diff(r) > 0)


def test_implied_real_curve_reproduces_published_tips(model):
    """A round-trip, not a validation.

    Breakeven inflation was *defined* as the ratio of the two published
    curves, so recovering the TIPS curve from the nominal curve and those
    breakevens is arithmetic.  Asserted to catch regressions in the Fisher
    decomposition.
    """
    t = np.array([5.0, 7.0, 10.0, 20.0, 30.0])
    assert model.real_rates.at(t) == pytest.approx(REAL_CURVE_2026.at(t), abs=2e-4)


def test_nominal_curve_round_trips(model):
    t = np.array([1.0, 5.0, 10.0, 30.0])
    assert model.nominal_rates(t) == pytest.approx(NOMINAL_CURVE_2026.at(t))


def test_inflation_expectations_are_anchored():
    """Breakevens sit near 2% at every horizon -- unlike 2012's upward slope."""
    r = INFLATION_BREAKEVEN_2026.at(np.array([1.0, 5.0, 10.0, 30.0]))
    assert np.all((r > 0.018) & (r < 0.025))
    assert r.max() - r.min() < 0.005


# -- debasement ---------------------------------------------------------------


def test_cf1_debasement(model, debt):
    """The paper's counterfactual-1 shift applied to today's debt."""
    fall = model.debasement(debt, counterfactual_cf1_2026()) / GDP_2026
    assert fall == pytest.approx(0.052, abs=0.005)


def test_debasement_scales_with_the_debt_not_the_maturity(model, debt):
    """Why 2026 differs from 2012, quantitatively.

    Duration is essentially unchanged between the two dates, so the whole
    increase in the potency of inflation comes from there being more debt.
    The ratio of debasement per point of inflation should track the ratio of
    market value to GDP, not anything about maturity.
    """
    from debtmodel.calibration import (
        GDP_2012,
        baseline_model,
        private_holdings,
    )

    m12, d12 = baseline_model(), private_holdings()
    fall12 = m12.debasement(d12, m12.shift_inflation(0.01)) / GDP_2012
    fall26 = model.debasement(debt, model.shift_inflation(0.01)) / GDP_2026

    duration_ratio = model.duration(debt) / m12.duration(d12)
    value_ratio = (model.value(debt) / GDP_2026) / (m12.value(d12) / GDP_2012)

    assert duration_ratio == pytest.approx(1.0, abs=0.1)
    assert fall26 / fall12 == pytest.approx(value_ratio, rel=0.1)


def test_a_point_of_inflation_still_buys_little(model, debt):
    """The paper's conclusion survives a 42%-larger debt stock.

    A permanent extra point of inflation at every horizon erodes under 3% of
    GDP, against a debt of 73% of GDP.
    """
    fall = model.debasement(debt, model.shift_inflation(0.01)) / GDP_2026
    assert 0.02 < fall < 0.03
    assert fall < 0.05 * model.value(debt) / GDP_2026


def test_duration_heuristic_holds(model, debt):
    """Erosion per point of inflation is about duration x value."""
    fall = model.debasement(debt, model.shift_inflation(0.01)) / GDP_2026
    heuristic = model.duration(debt) * model.value(debt) / GDP_2026 * 0.01
    assert fall == pytest.approx(heuristic, rel=0.15)


def test_short_end_is_still_where_the_money_is(model, debt):
    """Bills and near notes hold most of the value and almost none of the risk."""
    short = debt.truncated(4.5)
    assert model.value(short) / model.value(debt) > 0.6
    cf = counterfactual_cf1_2026()
    assert model.debasement(short, cf) / model.debasement(debt, cf) < 0.45


def test_repression_still_dominates_inflation(model, debt):
    """Section 8's lever is unchanged in kind, and larger in size."""
    cf = counterfactual_cf1_2026()
    plain = model.debasement(debt, cf)
    r = model.repression(debt, cf, 10)
    assert r.from_inflation > 1.5 * plain
    assert r.total / GDP_2026 > 0.20


def test_higher_inflation_monotonically_lowers_the_burden(model, debt):
    prev = model.value(debt)
    for extra in (0.01, 0.02, 0.05):
        v = model.shift_inflation(extra).value(debt)
        assert v < prev
        prev = v


def test_2012_parameters_are_untouched():
    """The 2026 set is additive; the replication must not have moved."""
    from debtmodel.calibration import (
        GDP_2012,
        INFLATION_BASELINE,
        baseline_model,
        counterfactual_model,
        private_holdings,
    )

    assert GDP_2012 == 16_200.0
    assert INFLATION_BASELINE.at(30.0) == pytest.approx(0.030)
    m, d = baseline_model(), private_holdings()
    assert m.value(d) / GDP_2012 == pytest.approx(0.512, abs=0.001)
    assert m.debasement(d, counterfactual_model(1)) / GDP_2012 == pytest.approx(
        0.037, abs=0.002
    )
