"""Checks on the debt-dynamics projections.

The baseline is validated against CBO's February 2026 published projections;
the scenarios are checked for the properties they are meant to illustrate.
"""

import numpy as np
import pytest

from debtmodel.fiscal import Assumptions, DebtLadder, project
from debtmodel.fiscal_calibration import (
    CBO_TARGETS,
    DEBT_HELD_BY_PUBLIC,
    GDP_2026,
    baseline,
    run,
    starting_ladder,
)
from debtmodel.scenarios import GROUPS, SCENARIOS, required_primary_balance, run_all


@pytest.fixture(scope="module")
def results():
    return run_all()


# -- the baseline against CBO -------------------------------------------------


def test_baseline_matches_cbo_2036():
    """CBO: debt 120% of GDP, deficit 6.7%, net interest 4.6% in 2036."""
    p = run(baseline(), years=11)
    t = CBO_TARGETS[2036]
    assert p.debt_gdp[-1] == pytest.approx(t["debt"], abs=0.015)
    assert p.interest_gdp[-1] == pytest.approx(t["interest"], abs=0.004)
    assert p.deficit_gdp[-1] == pytest.approx(t["deficit"], abs=0.005)


def test_baseline_starts_where_the_data_says():
    p = run(baseline(), years=2)
    t = CBO_TARGETS[2026]
    assert p.debt_gdp[0] == pytest.approx(t["debt"], abs=0.001)
    assert p.interest_gdp[0] == pytest.approx(t["interest"], abs=0.001)
    assert p.deficit_gdp[0] == pytest.approx(t["deficit"], abs=0.002)


def test_starting_ladder_reproduces_the_stock_and_rate():
    ladder = starting_ladder()
    assert ladder.total == pytest.approx(DEBT_HELD_BY_PUBLIC)
    assert ladder.effective_rate == pytest.approx(0.033 / 1.01, rel=1e-6)


def test_effective_rate_rises_toward_the_market_rate():
    """The budget feels rate changes with a lag set by the maturity ladder."""
    p = run(baseline(), years=15)
    assert p.effective_rate[0] < p.effective_rate[-1]
    assert p.effective_rate[-1] < p.market_rate[-1] + 0.015


def test_baseline_does_not_stabilise():
    assert not run(baseline(), years=25).stabilises()


# -- structural properties ----------------------------------------------------


def test_higher_growth_lowers_the_ratio():
    lo = run(baseline(real_growth=0.010), years=20).terminal_debt
    hi = run(baseline(real_growth=0.030), years=20).terminal_debt
    assert hi < lo


def test_larger_primary_surplus_lowers_the_ratio():
    a = run(baseline(primary_balance=-0.04), years=20).terminal_debt
    b = run(baseline(primary_balance=0.02), years=20).terminal_debt
    assert b < a


def test_repression_lowers_the_ratio():
    a = run(baseline(), years=20).terminal_debt
    b = run(baseline(repression_bp=200.0), years=20).terminal_debt
    assert b < a - 0.15


def test_risk_premium_is_self_reinforcing():
    """With a debt-sensitive premium the trajectory must steepen, not shift."""
    flat = run(baseline(), years=25)
    spiral = run(baseline(risk_premium_per_10pp=0.0025), years=25)
    gap = spiral.debt_gdp - flat.debt_gdp
    assert np.all(np.diff(gap) >= -1e-9)
    assert gap[-1] > 5 * gap[10]


def test_indexed_debt_is_not_eroded_by_inflation():
    """TIPS cost more when inflation is higher, which is the point of them."""
    a = run(baseline(inflation=0.043, indexed_share=0.0), years=15).terminal_debt
    b = run(baseline(inflation=0.043, indexed_share=0.20), years=15).terminal_debt
    assert b > a


def test_anticipated_inflation_helps_only_while_old_debt_rolls_off():
    """The gain is a transition effect, not a permanent change in dynamics.

    Higher inflation raises nominal GDP at once but reaches the interest bill
    only as debt matures.  So the ratio improves for roughly the length of the
    maturity structure and then resumes its baseline slope.
    """
    base = run(baseline(), years=30)
    infl = run(baseline(inflation=0.043), years=30)
    gain = base.debt_gdp - infl.debt_gdp
    assert gain[8] > gain[3]                       # builds while debt rolls
    early = gain[8] - gain[3]
    late = gain[-1] - gain[-6]
    assert late < early                            # then stops widening


def test_surprise_inflation_matches_the_stock_model():
    """The one-off haircut is the debasement number from the pricing model.

    calibration_2026 puts a permanent +1pp of inflation at 2.6% of GDP in
    present value; entered here as a year-one write-down it should move the
    ratio by about that much immediately.
    """
    base = run(baseline(), years=5)
    cut = run(baseline(haircut_gdp=0.026), years=5)
    assert base.debt_gdp[0] - cut.debt_gdp[0] == pytest.approx(0.026, abs=0.001)


# -- issuance policy ----------------------------------------------------------


def test_long_debt_only_protects_against_surprises(results):
    """The central claim in the debt-management debate, made precise.

    If a rate rise is foreseen it is priced into long coupons on issue, so
    maturity does not help.  Only an unanticipated rise rewards having
    borrowed long.
    """
    foreseen_long = run(baseline(real_short_rate=0.030), years=25).terminal_debt
    foreseen_bills = run(
        baseline(real_short_rate=0.030, issuance_shares=SCENARIOS["twist"].issuance_shares),
        years=25,
    ).terminal_debt
    assert foreseen_bills < foreseen_long            # no protection when priced in

    surprise_long = results["spike_long_funded"].terminal_debt
    surprise_bills = results["spike_bill_funded"].terminal_debt
    assert surprise_bills > surprise_long            # protection when it is not


def test_twist_is_a_two_sided_bet(results):
    """It wins if rates follow the forecast and loses if they spike."""
    base = results["baseline"].terminal_debt
    assert results["twist"].terminal_debt < base
    assert (
        results["spike_bill_funded"].terminal_debt
        > results["spike_long_funded"].terminal_debt
    )


# -- the scenario groups do what they claim -----------------------------------


def test_worse_scenarios_are_worse(results):
    base = results["baseline"].terminal_debt
    for name in GROUPS["Makes it worse"]:
        assert results[name].terminal_debt > base, name


def test_stabilising_scenarios_flatten_or_beat_the_baseline(results):
    base = results["baseline"].terminal_debt
    for name in GROUPS["Stabilises"]:
        assert results[name].terminal_debt < base - 0.20, name


def test_reducing_scenarios_actually_reduce(results):
    for name in ["postwar", "grand_bargain"]:
        p = results[name]
        assert p.terminal_debt < p.debt_gdp[0], name
        assert p.stabilises(), name


def test_default_does_not_solve_anything(results):
    """A 30% write-down is undone within a decade once rates and growth react."""
    p = results["default"]
    assert p.debt_gdp[0] < results["baseline"].debt_gdp[0]     # helps at once
    assert p.terminal_debt > results["baseline"].terminal_debt  # and then does not


# -- what it would take -------------------------------------------------------


def test_required_primary_balance_is_reachable_but_far():
    """Stabilising needs roughly a 2pp swing from the current primary deficit."""
    need = required_primary_balance()
    assert -0.02 < need < 0.02
    gap = need - (-0.025)
    assert 0.015 < gap < 0.035


def test_repression_does_much_of_the_work():
    """175bp of repression removes more than half the required adjustment."""
    plain = required_primary_balance()
    repressed = required_primary_balance(repression_bp=175.0)
    assert repressed < plain - 0.010


def test_worse_conditions_raise_the_required_adjustment():
    assert required_primary_balance(real_short_rate=0.022) > required_primary_balance()
    assert required_primary_balance(real_growth=0.010) > required_primary_balance()


# -- engine mechanics ---------------------------------------------------------


def test_ladder_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        DebtLadder(np.ones(30), np.ones(29))


def test_zero_deficit_and_zero_rate_holds_debt_flat_in_nominal_terms():
    """With no growth, no interest and no primary gap, the ratio cannot move.

    Issuance is confined to one year so that the term premium -- which is
    positive even when the short rate is zero -- does not creep into the
    coupon and quietly reintroduce a deficit.
    """
    ladder = DebtLadder(np.full(30, 100.0), np.zeros(30))
    bills_only = np.zeros(30)
    bills_only[0] = 1.0
    a = Assumptions(
        real_growth=0.0, inflation=0.0, real_short_rate=0.0,
        primary_balance=0.0, indexed_share=0.0, issuance_shares=bills_only,
    )
    p = project(ladder, 10_000.0, a, years=10)
    assert p.debt_gdp == pytest.approx(np.full(10, 0.3))


def test_term_premium_costs_money_even_at_a_zero_short_rate():
    """Borrowing long is not free when the short rate is zero."""
    ladder = DebtLadder(np.full(30, 100.0), np.zeros(30))
    long_only = np.zeros(30)
    long_only[29] = 1.0
    a = Assumptions(
        real_growth=0.0, inflation=0.0, real_short_rate=0.0,
        primary_balance=0.0, indexed_share=0.0, issuance_shares=long_only,
    )
    p = project(ladder, 10_000.0, a, years=10)
    assert p.debt_gdp[-1] > p.debt_gdp[0]


def test_primary_surplus_can_retire_debt():
    p = run(baseline(primary_balance=0.08), years=20)
    assert p.terminal_debt < 0.30
