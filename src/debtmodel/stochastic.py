"""Monte Carlo around a scenario.

The deterministic projections answer "where does this path lead?". They cannot
answer "how likely is it to work?", which is the question a policy actually
faces. A consolidation that stabilises the debt in the central case may still
fail most of the time if growth or rates go against it.

The shock structure is deliberately plain: three correlated AR(1) processes on
real growth, inflation and the real short rate, plus an explicit recession
regime, because Gaussian noise alone does not produce the clustered, asymmetric
downturns that dominate real fiscal outcomes. A 2008 is not three standard
deviations of a normal; it is a different state of the world.

Everything is layered on top of a scenario's own assumptions, so the same
policy can be run deterministically and stochastically and compared.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .fiscal import Assumptions, Projection, project
from .fiscal_calibration import GDP_2026, starting_ladder

__all__ = ["ShockProcess", "FanChart", "simulate"]


@dataclass(frozen=True)
class ShockProcess:
    """Calibration of the shocks applied on top of a scenario's central path.

    Standard deviations are annual, in decimals. Defaults are set to postwar
    US experience: real growth has a standard deviation near 2 percentage
    points, inflation somewhat less and highly persistent, the real short rate
    persistent and slower-moving.
    """

    growth_sd: float = 0.018
    growth_rho: float = 0.30
    inflation_sd: float = 0.010
    inflation_rho: float = 0.65
    rate_sd: float = 0.008
    rate_rho: float = 0.80

    #: Correlations between the three innovations. Growth and rates move
    #: together because policy leans against the cycle; inflation and rates
    #: move together through the same channel.
    corr_growth_rate: float = 0.30
    corr_inflation_rate: float = 0.40
    corr_growth_inflation: float = 0.20

    #: Annual probability of entering a recession, and its severity. Roughly
    #: one recession per decade, costing about three points of growth in the
    #: first year and widening the primary deficit by two points through the
    #: automatic stabilisers.
    recession_prob: float = 0.10
    recession_growth_hit: float = 0.030
    recession_deficit_hit: float = 0.020
    recession_length: int = 2

    def _chol(self) -> np.ndarray:
        c = np.array([
            [1.0, self.corr_growth_inflation, self.corr_growth_rate],
            [self.corr_growth_inflation, 1.0, self.corr_inflation_rate],
            [self.corr_growth_rate, self.corr_inflation_rate, 1.0],
        ])
        return np.linalg.cholesky(c)


@dataclass(frozen=True)
class FanChart:
    """Percentile bands and outcome probabilities across simulated paths."""

    label: str
    years: np.ndarray
    percentiles: dict[int, np.ndarray]
    paths_terminal: np.ndarray
    central: np.ndarray

    def p(self, q: int) -> np.ndarray:
        return self.percentiles[q]

    def prob_below(self, level: float) -> float:
        """Share of paths ending at or below ``level`` (as a share of GDP)."""
        return float((self.paths_terminal <= level).mean())

    def prob_above(self, level: float) -> float:
        return float((self.paths_terminal > level).mean())

    @property
    def median_terminal(self) -> float:
        return float(np.median(self.paths_terminal))


def _draw(shocks: ShockProcess, years: int, n: int, rng: np.random.Generator):
    """Correlated AR(1) paths for the three drivers, plus recession flags."""
    L = shocks._chol()
    sd = np.array([shocks.growth_sd, shocks.inflation_sd, shocks.rate_sd])
    rho = np.array([shocks.growth_rho, shocks.inflation_rho, shocks.rate_rho])

    eps = rng.standard_normal((n, years, 3)) @ L.T
    eps = eps * sd * np.sqrt(1.0 - rho ** 2)   # scale so the stationary sd is sd

    state = np.zeros((n, years, 3))
    prev = rng.standard_normal((n, 3)) * sd    # start from the stationary spread
    for t in range(years):
        prev = rho * prev + eps[:, t, :]
        state[:, t, :] = prev

    # Recessions: an independent hazard each year, persisting for a few years.
    hit = rng.random((n, years)) < shocks.recession_prob
    rec = np.zeros((n, years))
    for t in range(years):
        for k in range(shocks.recession_length):
            if t + k < years:
                rec[:, t + k] = np.maximum(
                    rec[:, t + k], hit[:, t] * (1.0 - k / shocks.recession_length)
                )
    return state, rec


def simulate(
    assumptions: Assumptions,
    years: int = 25,
    n: int = 2000,
    shocks: ShockProcess | None = None,
    seed: int = 20260908,
    quantiles: tuple[int, ...] = (5, 25, 50, 75, 95),
) -> FanChart:
    """Run ``n`` shocked versions of one scenario and collect the distribution.

    The scenario's own assumptions supply the central path; shocks are added to
    it, so a consolidation scenario stays a consolidation scenario in every
    draw. Expectations are held at the *central* path throughout, which means
    every shock is a surprise -- the realistic case, and the one in which the
    maturity structure matters.
    """
    rng = np.random.default_rng(seed)
    shocks = shocks or ShockProcess()
    ladder = starting_ladder()

    from .fiscal import _as_path

    g0 = _as_path(assumptions.real_growth, years)
    pi0 = _as_path(assumptions.inflation, years)
    rr0 = _as_path(assumptions.real_short_rate, years)
    pb0 = _as_path(assumptions.primary_balance, years)
    expected = (
        rr0 if assumptions.expected_real_short_rate is None
        else _as_path(assumptions.expected_real_short_rate, years)
    )

    state, rec = _draw(shocks, years, n, rng)
    out = np.zeros((n, years))

    for i in range(n):
        a = replace(
            assumptions,
            real_growth=np.maximum(
                g0 + state[i, :, 0] - rec[i] * shocks.recession_growth_hit, -0.08
            ),
            inflation=np.maximum(pi0 + state[i, :, 1], -0.02),
            real_short_rate=rr0 + state[i, :, 2],
            # Shocks are unanticipated: bonds are priced off the central path.
            expected_real_short_rate=expected,
            primary_balance=pb0 - rec[i] * shocks.recession_deficit_hit,
        )
        out[i] = project(ladder, GDP_2026, a, years=years).debt_gdp

    central = project(ladder, GDP_2026, assumptions, years=years).debt_gdp
    return FanChart(
        label=assumptions.label,
        years=np.arange(2026, 2026 + years),
        percentiles={q: np.percentile(out, q, axis=0) for q in quantiles},
        paths_terminal=out[:, -1],
        central=central,
    )
