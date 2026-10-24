"""Checks on the Monte Carlo layer."""

import numpy as np
import pytest

from debtmodel.fiscal_calibration import baseline, run
from debtmodel.primary_balance import PACKAGES, package
from debtmodel.stochastic import ShockProcess, simulate

N = 300          # small but stable enough for the properties asserted here


@pytest.fixture(scope="module")
def fan():
    return simulate(baseline(), n=N)


def test_percentiles_are_ordered(fan):
    for t in range(len(fan.years)):
        vals = [fan.p(q)[t] for q in (5, 25, 50, 75, 95)]
        assert vals == sorted(vals)


def test_uncertainty_widens_with_horizon(fan):
    early = fan.p(95)[3] - fan.p(5)[3]
    late = fan.p(95)[-1] - fan.p(5)[-1]
    assert late > 3 * early


def test_all_paths_start_at_the_same_place(fan):
    assert fan.p(5)[0] == pytest.approx(fan.p(95)[0], abs=0.02)
    assert fan.p(50)[0] == pytest.approx(1.01, abs=0.02)


def test_zero_shocks_reproduce_the_deterministic_path():
    """With every variance and the recession hazard set to zero, the simulation
    must collapse onto the scenario it was built from."""
    quiet = ShockProcess(
        growth_sd=0.0, inflation_sd=0.0, rate_sd=0.0, recession_prob=0.0
    )
    f = simulate(baseline(), n=8, shocks=quiet)
    assert f.p(50) == pytest.approx(f.central, abs=1e-9)
    assert f.paths_terminal.std() == pytest.approx(0.0, abs=1e-9)


def test_the_median_is_worse_than_the_central_path(fan):
    """Debt dynamics are convex and recessions are one-sided.

    A bad draw raises debt, which raises interest, which raises debt again;
    a good draw does not compound symmetrically. Adding a realistic recession
    hazard on top means the median outcome sits well above a projection that
    contains no recessions at all -- which is what a CBO-style baseline is.
    """
    assert fan.median_terminal > fan.central[-1] + 0.05


def test_recessions_are_what_drives_that_gap():
    no_rec = simulate(baseline(), n=N, shocks=ShockProcess(recession_prob=0.0))
    with_rec = simulate(baseline(), n=N, shocks=ShockProcess(recession_prob=0.10))
    assert with_rec.median_terminal > no_rec.median_terminal + 0.05


def test_probabilities_are_consistent(fan):
    assert fan.prob_below(1.20) + fan.prob_above(1.20) == pytest.approx(1.0)
    assert 0.0 <= fan.prob_below(1.01) <= 1.0
    assert fan.prob_below(3.0) > fan.prob_below(1.5)


def test_consolidation_raises_the_odds_of_stabilising():
    """The point of running this stochastically."""
    base = simulate(baseline(), n=N)
    cons = simulate(package(0.0254, 0), n=N)
    assert cons.prob_below(1.20) > base.prob_below(1.20) + 0.3
    assert cons.median_terminal < base.median_terminal


def test_even_a_stabilising_package_is_not_a_guarantee():
    """It works in the central case and still fails a meaningful share of the
    time -- the honest reading of a point projection."""
    f = simulate(package(0.0254, 0), n=N)
    assert f.prob_above(1.10) > 0.10


def test_bigger_shocks_widen_the_fan():
    calm = simulate(baseline(), n=N, shocks=ShockProcess(growth_sd=0.005))
    wild = simulate(baseline(), n=N, shocks=ShockProcess(growth_sd=0.035))
    calm_w = calm.p(95)[-1] - calm.p(5)[-1]
    wild_w = wild.p(95)[-1] - wild.p(5)[-1]
    assert wild_w > calm_w


def test_repression_narrows_the_fan_as_well_as_lowering_it():
    """Repression fixes part of the interest bill by decree, so rate shocks
    pass through less. It buys certainty as well as level."""
    plain = simulate(package(0.0254, 0), n=N)
    repressed = simulate(package(0.0, 300), n=N)
    assert (repressed.p(95)[-1] - repressed.p(5)[-1]) < (
        plain.p(95)[-1] - plain.p(5)[-1]
    )


def test_simulation_is_reproducible():
    a = simulate(baseline(), n=40, seed=7).paths_terminal
    b = simulate(baseline(), n=40, seed=7).paths_terminal
    assert a == pytest.approx(b)
    c = simulate(baseline(), n=40, seed=8).paths_terminal
    assert not np.allclose(a, c)
