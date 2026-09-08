"""Check the model against numbers published in Hilscher, Raviv & Reis (2014).

These are the tests that matter: they compare against the paper, not against
the model's own output.  Tolerances are loose where the paper's input is a
figure that had to be read off a chart.
"""

import numpy as np
import pytest

from debtmodel import DebasementModel, TermStructure
from debtmodel.calibration import (
    GDP_2012,
    INFLATION_BASELINE,
    NOMINAL_CURVE_2012,
    PAPER_TABLE_3_PRIVATE,
    baseline_model,
    counterfactual_model,
    private_holdings,
)


@pytest.fixture(scope="module")
def profile():
    return private_holdings()


@pytest.fixture(scope="module")
def model():
    return baseline_model()


# -- section 3: the debt data ------------------------------------------------


def test_market_value_matches_paper(model, profile):
    """Section 3.1: privately-held marketable nominals were $8.3tn, 51.2% of GDP."""
    assert model.value(profile) / GDP_2012 == pytest.approx(0.512, abs=0.001)
    assert model.value(profile) == pytest.approx(8_300.0, rel=0.01)


def test_average_maturity_matches_paper(profile):
    """Section 3.2: average maturity weighted by private holdings is 4.8 years."""
    assert profile.average_maturity == pytest.approx(4.8, abs=0.1)


def test_duration_is_shorter_than_average_maturity(model, profile):
    """Discounting front-loads the weights, so Macaulay duration is shorter."""
    assert model.duration(profile) < profile.average_maturity


# -- section 4: the implied real curve ---------------------------------------


def test_implied_real_rates_are_negative_and_upward_sloping(model):
    """End-2012 TIPS yields were about -1.4% at 5y, -0.7% at 10y, ~+0.4% at 30y.

    Nothing in the calibration targets these -- they fall out of the observed
    nominal curve and the paper's table 4 inflation -- so agreement is a real
    check that the two sources are mutually consistent.
    """
    r = model.real_rates.at(np.array([5.0, 10.0, 30.0]))
    assert r[0] == pytest.approx(-0.014, abs=0.005)
    assert r[1] == pytest.approx(-0.007, abs=0.005)
    assert r[2] == pytest.approx(0.004, abs=0.006)
    assert np.all(np.diff(r) > 0)


def test_nominal_rates_round_trip(model):
    """Fisher decomposition and recomposition are inverses."""
    t = np.array([1.0, 5.0, 10.0, 30.0])
    assert model.nominal_rates(t) == pytest.approx(NOMINAL_CURVE_2012.at(t))


# -- section 7: the counterfactuals ------------------------------------------


def test_counterfactual_1_matches_table_3(model, profile):
    """Table 3 row 1: permanently higher inflation lowers debt by 3.7% of GDP."""
    fall = model.debasement(profile, counterfactual_model(1)) / GDP_2012
    assert fall == pytest.approx(0.037, abs=0.002)


@pytest.mark.parametrize("n", sorted(PAPER_TABLE_3_PRIVATE))
def test_all_counterfactuals_are_in_the_right_neighbourhood(model, profile, n):
    """Every row of table 3, to within half a percentage point of GDP.

    Row 3 ("higher and more variable") is the one this model structurally
    cannot get right: it differs from row 1 only in the variance of inflation,
    which a deterministic model does not see.  See
    ``test_deterministic_model_cannot_price_inflation_uncertainty``.
    """
    fall = model.debasement(profile, counterfactual_model(n)) / GDP_2012
    assert fall == pytest.approx(PAPER_TABLE_3_PRIVATE[n], abs=0.005)


def test_maturity_truncations_match_table_5(model, profile):
    """Table 5: restricting counterfactual 1 to short maturities gives 0.1% and 1.1%."""
    cf = counterfactual_model(1)
    within_1y = model.debasement(profile.truncated(1.0), cf) / GDP_2012
    within_45y = model.debasement(profile.truncated(4.5), cf) / GDP_2012
    assert within_1y == pytest.approx(0.001, abs=0.002)
    assert within_45y == pytest.approx(0.011, abs=0.002)


def test_short_maturity_debt_is_nearly_immune(model, profile):
    """Section 7.5's central claim: debasement lives almost entirely at the long end."""
    cf = counterfactual_model(1)
    total = model.debasement(profile, cf)
    short = model.debasement(profile.truncated(4.5), cf)
    # Under half the debasement comes from the maturities holding most of the value.
    assert short / total < 0.5
    assert model.value(profile.truncated(4.5)) / model.value(profile) > 0.6


# -- section 8: financial repression -----------------------------------------


def test_repression_alone_lowers_the_burden(model, profile):
    """Table 6 column 1: repression with no extra inflation still erodes debt.

    It works through positive nominal rates -- deferring a payment at zero
    interest is a loss to the holder whenever the nominal rate is positive.
    """
    for years in (1, 5, 10):
        result = model.repression(profile, counterfactual_model(1), years)
        assert result.from_repression > 0.0


def test_repression_decomposition_sums(model, profile):
    """The two channels of table 6 add to its total column."""
    result = model.repression(profile, counterfactual_model(1), 5)
    assert result.from_repression + result.from_inflation == pytest.approx(result.total)


def test_repression_makes_inflation_more_effective(model, profile):
    """Section 8's conclusion: repression is what makes inflation bite.

    Ten years of repression roughly triples the debasement from the same
    inflation shift (paper: 3.7% -> 10.7% of GDP).
    """
    cf = counterfactual_model(1)
    plain = model.debasement(profile, cf)
    repressed = model.repression(profile, cf, 10).from_inflation
    assert repressed > 2.0 * plain


def test_repression_effects_are_increasing_in_duration(model, profile):
    cf = counterfactual_model(1)
    totals = [model.repression(profile, cf, n).total for n in (1, 5, 10)]
    assert totals == sorted(totals)


# -- structural properties and known limitations -----------------------------


def test_weights_decline_with_maturity(model):
    """Payments further out are worth less: omega_t is decreasing."""
    w = model.weights(np.arange(1.0, 31.0))
    assert np.all(np.diff(w) < 0)


def test_zero_inflation_and_zero_real_rates_give_face_value(profile):
    flat = DebasementModel(
        real_rates=TermStructure.flat(0.0), inflation=TermStructure.flat(0.0)
    )
    assert flat.value(profile) == pytest.approx(profile.face_value)


def test_higher_inflation_always_lowers_the_burden(model, profile):
    """Monotonicity: with real rates fixed, more inflation cannot raise W."""
    prev = model.value(profile)
    for extra in (0.01, 0.02, 0.05, 0.10):
        v = model.shift_inflation(extra).value(profile)
        assert v < prev
        prev = v


def test_immediate_payments_cannot_be_inflated_away():
    """A payment due now is untouched by any future inflation."""
    from debtmodel import DebtProfile

    now = DebtProfile(np.array([100.0]), np.array([0.0]), label="due now")
    model = baseline_model()
    assert model.debasement(now, model.shift_inflation(0.5)) == pytest.approx(0.0)


def test_deterministic_model_cannot_price_inflation_uncertainty(model, profile):
    """A documented limitation, not a bug.

    Counterfactuals 1 and 3 have nearly identical mean inflation but very
    different variance; the paper separates them (3.7% vs 3.4% of GDP) because
    E[1/pi] > 1/E[pi].  A fixed-parameter model sees only the mean, so it
    collapses them.  Recovering the gap needs the risk-adjusted density.
    """
    d1 = model.debasement(profile, counterfactual_model(1))
    d3 = model.debasement(profile, counterfactual_model(3))
    assert abs(d1 - d3) / GDP_2012 < 0.002
    assert abs(PAPER_TABLE_3_PRIVATE[1] - PAPER_TABLE_3_PRIVATE[3]) == pytest.approx(
        0.003, abs=1e-9
    )


def test_single_bond_approximation_is_badly_wrong(model, profile):
    """Why the model keeps the full profile.

    The paper's opening move is that treating the debt as one bond at the
    average maturity gives 5.5% for a 1pp shift.  Here the same shortcut,
    applied to counterfactual 1, overstates the true answer substantially.
    """
    from debtmodel import DebtProfile

    cf = counterfactual_model(1)
    true = model.debasement(profile, cf)
    single = DebtProfile(
        np.array([model.value(profile)]),
        np.array([profile.average_maturity]),
        label="one bond at average maturity",
    )
    # Re-par the single bond so it has the same market value, then reprice.
    single = single.scaled(model.value(profile) / model.value(single))
    assert model.debasement(single, cf) > 1.2 * true


# -- input validation --------------------------------------------------------


def test_rejects_decreasing_tenors():
    with pytest.raises(ValueError, match="strictly increasing"):
        TermStructure(np.array([5.0, 1.0]), np.array([0.01, 0.02]))


def test_rejects_negative_payments():
    from debtmodel import DebtProfile

    with pytest.raises(ValueError, match="non-negative"):
        DebtProfile(np.array([100.0, -1.0]))


def test_rejects_unknown_counterfactual():
    with pytest.raises(KeyError):
        counterfactual_model(99)
