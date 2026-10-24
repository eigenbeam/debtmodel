"""Checks on the primary-balance scenarios."""

import numpy as np
import pytest

from debtmodel.fiscal_calibration import baseline, run
from debtmodel.primary_balance import (
    BUDGET_2026,
    DELAYS,
    HISTORICAL,
    MULTIPLIER,
    SIZES,
    START_BALANCE,
    TARGETS,
    budget_equivalents,
    consolidation,
    cost_of_delay,
    entitlement_drift,
    path,
    required_for_target,
)


# -- the path builder ---------------------------------------------------------


def test_path_starts_at_todays_balance_and_reaches_the_target():
    p = path(0.028, over=6, start=1)
    assert p[0] == pytest.approx(START_BALANCE)
    assert p[6] == pytest.approx(START_BALANCE + 0.028)
    assert p[-1] == pytest.approx(START_BALANCE + 0.028)


def test_path_phases_in_monotonically():
    p = path(0.028, over=6)
    assert np.all(np.diff(p) >= -1e-12)


def test_delay_holds_the_balance_flat_first():
    p = path(0.028, over=6, start=5)
    assert np.all(p[:5] == pytest.approx(START_BALANCE))
    assert p[5] > START_BALANCE


def test_lapsing_consolidation_returns_to_the_starting_balance():
    p = path(0.028, over=6, lapses_after=10)
    assert p[9] == pytest.approx(START_BALANCE + 0.028)
    assert p[-1] == pytest.approx(START_BALANCE)


# -- size, speed, delay -------------------------------------------------------


def test_bigger_swings_lower_the_ratio_monotonically():
    ends = [run(consolidation(s), years=25).terminal_debt for s in SIZES]
    assert ends == sorted(ends, reverse=True)


def test_faster_is_better_but_the_gap_is_modest():
    """Speed helps, but far less than size does.

    Phasing the same 2.8 points over 15 years instead of 2 costs about 17
    points of GDP by 2050 -- real, but a fifth of what halving the size costs.
    """
    fast = run(consolidation(0.028, over=2), years=25).terminal_debt
    slow = run(consolidation(0.028, over=15), years=25).terminal_debt
    assert fast < slow
    assert (slow - fast) < 0.25


def test_delay_is_the_expensive_dimension():
    """Each year of waiting raises the swing needed to reach the same place."""
    sizes = [s for _, s in cost_of_delay()]
    assert sizes == sorted(sizes)
    assert sizes[-1] > 3 * sizes[0]


def test_waiting_fifteen_years_exceeds_any_historical_adjustment():
    late = dict(cost_of_delay())[15]
    assert late > max(h["swing"] for h in HISTORICAL[:2])


def test_a_lapsing_consolidation_does_not_stick():
    lasting = run(consolidation(0.028), years=25).terminal_debt
    lapsing = run(
        baseline(primary_balance=path(0.028, lapses_after=10),
                 fiscal_multiplier=MULTIPLIER),
        years=25,
    ).terminal_debt
    assert lapsing > lasting + 0.20


# -- the fiscal multiplier ----------------------------------------------------


def test_multiplier_makes_consolidation_harder_but_not_self_defeating():
    """Austerity does not raise the debt ratio here, at a phased pace.

    A large multiplier with full hysteresis would be needed for that, and even
    then the offset is partial. The claim being checked is the weaker and more
    defensible one: ignoring the multiplier flatters the result, modestly.
    """
    without = run(consolidation(0.028, multiplier=0.0), years=25).terminal_debt
    with_mult = run(consolidation(0.028, multiplier=MULTIPLIER), years=25).terminal_debt
    big = run(consolidation(0.028, multiplier=1.5), years=25).terminal_debt
    assert without < with_mult < big
    assert big - without < 0.05
    assert big < run(baseline(), years=25).terminal_debt


def test_phasing_blunts_the_annual_growth_hit_not_the_cumulative_one():
    """Speed changes when the growth cost lands, not how much of it there is.

    The hit in any one year is the multiplier times that year's impulse, so
    phasing the same swing over 15 years instead of 2 shrinks the worst year
    by a factor of about seven. The permanent level loss, however, is the
    multiplier times the *total* swing times hysteresis either way -- which is
    why the 2050 ratios end up close regardless of pace.
    """
    def worst_impulse(over):
        return float(np.max(np.diff(path(0.028, over=over))))

    assert worst_impulse(2) > 6 * worst_impulse(15)

    def gap(over):
        a = run(consolidation(0.028, over=over, multiplier=0.0), years=25).terminal_debt
        b = run(consolidation(0.028, over=over, multiplier=1.5), years=25).terminal_debt
        return b - a

    # Cumulative cost is of a similar order at either pace.
    assert abs(gap(15) - gap(2)) < 0.02


# -- targets ------------------------------------------------------------------


def test_required_swing_for_each_target_is_ordered():
    sizes = [required_for_target(t) for t in TARGETS.values()]
    assert sizes == sorted(sizes)


def test_solved_swing_actually_hits_its_target():
    for t in TARGETS.values():
        s = required_for_target(t)
        assert run(consolidation(s), years=25).terminal_debt == pytest.approx(t, abs=0.01)


def test_stabilising_needs_roughly_two_and_a_half_points():
    assert required_for_target(1.01) == pytest.approx(0.025, abs=0.005)


# -- budget translation -------------------------------------------------------


def test_budget_equivalents_are_internally_consistent():
    e = budget_equivalents(0.028)
    assert e["$bn per year"] == pytest.approx(0.028 * 32_486.0)
    assert e["share of revenues"] == pytest.approx(0.028 / BUDGET_2026["revenues"])
    assert e["share of outlays"] < e["share of non-interest outlays"]


def test_a_stabilising_swing_is_about_a_sixth_of_revenue():
    e = budget_equivalents(required_for_target(1.01))
    assert 0.12 < e["share of revenues"] < 0.20


def test_entitlement_drift_is_most_of_a_stabilising_swing():
    """Social Security and Medicare growth alone nearly matches the target.

    The gross policy effort therefore exceeds the net improvement: holding the
    primary balance flat already requires offsetting this.
    """
    drift = entitlement_drift()
    assert drift == pytest.approx(0.019, abs=0.001)
    assert drift > 0.6 * required_for_target(1.01)


# -- consolidation combined with financial repression -------------------------


def test_repression_substitutes_for_consolidation_about_one_for_one():
    """With debt near 100% of GDP, a basis point off the rate is worth about a
    basis point of GDP in primary balance -- less the Fed's share, which
    repression cannot touch."""
    from debtmodel.primary_balance import EFFECTIVE_SHARE, repression_frontier

    f = repression_frontier(levels=(0, 100, 200))
    per_100bp = (f[0]["size"] - f[2]["size"]) / 2
    assert per_100bp == pytest.approx(0.01 * EFFECTIVE_SHARE, abs=0.002)


def test_the_frontier_is_monotone_and_roughly_linear():
    from debtmodel.primary_balance import repression_frontier

    f = repression_frontier()
    sizes = [r["size"] for r in f]
    assert sizes == sorted(sizes, reverse=True)
    steps = np.diff(sizes)
    assert np.std(steps) < 0.0005          # near-constant substitution rate


def test_repressing_the_fed_gains_nothing():
    """The Fed remits its income back, so a below-market rate on its holdings
    is neutral. Ignoring that would overstate repression by about a seventh."""
    from debtmodel.primary_balance import package

    honest = run(package(0.0, 300), years=25).terminal_debt
    overstated = run(
        package(0.0, 300, repression_effective_share=1.0), years=25
    ).terminal_debt
    assert overstated < honest


def test_every_package_reaches_the_same_destination():
    """The five packages differ in means, not in ends -- which is the point."""
    from debtmodel.primary_balance import PACKAGES, package

    for label, size, bp in PACKAGES:
        assert run(package(size, bp), years=25).terminal_debt == pytest.approx(
            1.01, abs=0.005
        ), label


def test_repression_is_not_cheaper_only_differently_borne():
    """The headline claim, made precise.

    Shifting from an all-explicit adjustment to an all-implicit one leaves the
    total cost essentially unchanged -- it rises slightly -- while cutting what
    residents pay by about a third, because roughly a third of US debt is held
    abroad.
    """
    from debtmodel.primary_balance import PACKAGES, domestic_burden

    all_cons = domestic_burden(PACKAGES[0][1], PACKAGES[0][2])
    all_repr = domestic_burden(PACKAGES[-1][1], PACKAGES[-1][2])

    assert all_repr["total, $bn"] >= all_cons["total, $bn"]
    assert all_repr["total, $bn"] / all_cons["total, $bn"] < 1.10
    assert all_repr["borne domestically, $bn"] < 0.75 * all_cons["borne domestically, $bn"]
    assert all_cons["borne abroad, $bn"] == 0.0


def test_domestic_burden_falls_monotonically_toward_repression():
    from debtmodel.primary_balance import PACKAGES, domestic_burden

    dom = [domestic_burden(s, bp)["borne domestically, $bn"] for _, s, bp in PACKAGES]
    assert dom == sorted(dom, reverse=True)


def test_transfer_arithmetic_excludes_the_fed():
    from debtmodel.primary_balance import (
        DEBT_HELD_BY_PUBLIC,
        FED_HOLDINGS,
        repression_transfer,
    )

    t = repression_transfer(100)
    assert t["$bn per year"] == pytest.approx(
        0.01 * (DEBT_HELD_BY_PUBLIC - FED_HOLDINGS)
    )
    assert t["from foreign holders, $bn"] + t["from domestic holders, $bn"] == (
        pytest.approx(t["$bn per year"])
    )
    assert 0.3 < t["exported share"] < 0.4


def test_consolidation_is_slightly_more_efficient_than_repression():
    """Cutting the deficit shrinks the stock that interest is charged on;
    repression only lowers the rate. So the all-repression package needs a
    marginally larger gross transfer to reach the same debt ratio."""
    from debtmodel.primary_balance import PACKAGES, domestic_burden

    assert (
        domestic_burden(*PACKAGES[-1][1:])["total, $bn"]
        > domestic_burden(*PACKAGES[0][1:])["total, $bn"]
    )


# -- Japan --------------------------------------------------------------------


def test_japan_carries_twice_the_debt_for_a_third_of_the_interest():
    """The single most useful comparison in the whole exercise.

    Japan's debt ratio is roughly double the US level while its interest bill
    is under a third of it, which is what sustained repression buys.
    """
    from debtmodel.primary_balance import JAPAN

    us_debt, us_interest = 1.01, 0.033
    assert JAPAN["gross debt, % of GDP"] / us_debt > 1.9
    assert JAPAN["interest, % of GDP"] / us_interest < 0.35
    assert JAPAN["implied average rate"] < 0.006


def test_japans_holder_base_is_overwhelmingly_captive():
    """Central bank, banks, insurers and pensions together, against a foreign
    share half the US level. That composition is the repression."""
    from debtmodel.primary_balance import JAPAN, US_HOLDERS

    captive = (
        JAPAN["Bank of Japan share of JGBs and T-bills"]
        + JAPAN["banks"] + JAPAN["insurers"] + JAPAN["public and private pensions"]
    )
    assert captive > 0.80
    assert JAPAN["foreign"] < US_HOLDERS["foreign"] / 2
    assert JAPAN["Bank of Japan share of JGBs and T-bills"] > (
        3 * US_HOLDERS["Federal Reserve share"]
    )
