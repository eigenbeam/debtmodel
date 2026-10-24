"""Checks that the practical measures are scaled correctly against the target."""

import pytest

from debtmodel.policy_options import (
    CUMULATIVE_GDP_10Y,
    REPRESSION,
    REVENUE,
    SPENDING,
    Option,
    as_share_of_gdp,
    largest_single_option,
    options_needed,
)
from debtmodel.primary_balance import required_for_target


def test_score_conversion_is_a_ten_year_average():
    assert as_share_of_gdp(CUMULATIVE_GDP_10Y) == pytest.approx(1.0)
    assert Option("x", 3_400).annual_bn == pytest.approx(340.0)


def test_the_largest_option_is_about_one_point_of_gdp():
    assert largest_single_option().share_of_gdp == pytest.approx(0.009, abs=0.002)


def test_stabilising_takes_several_of_the_largest_options():
    """The headline framing: this is not one difficult vote, it is three."""
    n = options_needed(required_for_target(1.01))
    assert 2.0 < n < 4.0


def test_no_single_scored_option_comes_close_on_its_own():
    need = required_for_target(1.01)
    assert all(o.share_of_gdp < need / 2 for o in REVENUE + SPENDING)


def test_revenue_options_are_larger_than_spending_options():
    """Not a claim about desirability -- a fact about what CBO can score.

    The big revenue options are broad-based new taxes; the big spending
    options are structural changes to health programmes, and they score
    smaller.
    """
    assert max(o.ten_year_bn for o in REVENUE) > max(o.ten_year_bn for o in SPENDING)


def test_every_option_carries_a_score_and_a_label():
    for o in REVENUE + SPENDING:
        assert o.label and o.ten_year_bn > 0


def test_repression_instruments_are_described_not_scored():
    """Repression has no CBO line, which is exactly why it is politically
    attractive and why it belongs in a different list."""
    assert len(REPRESSION) >= 4
    for name, how in REPRESSION:
        assert name and len(how) > 80
