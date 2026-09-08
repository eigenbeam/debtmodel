"""The debt burden and its debasement by inflation.

Proposition 1 of Hilscher, Raviv & Reis (2014) states that the real market
value of the outstanding nominal debt is a weighted sum of the nominal
payments due at every future date,

    W = sum_t omega_t * B_0^t,     omega_t = R_t^{-1} * E^Q[1 / pi_{0,t}]      (7)-(8)

where ``R_t`` is the gross real risk-free return between 0 and ``t`` and
``pi_{0,t} = P_t / P_0`` is gross cumulative inflation, under the risk-adjusted
(risk-neutral) measure.

This module implements the *fixed-parameter* case, in which inflation is
deterministic given a term structure of average annual rates, so the
expectation collapses:

    pi_{0,t} = (1 + pibar_t)^t,   R_t = (1 + r_t)^t,
    omega_t  = [(1 + r_t)(1 + pibar_t)]^{-t}

Two things are worth knowing about that simplification before relying on it.

It is cheap on the inflation side.  ``E^Q[1/pi]`` exceeds ``1/E^Q[pi]`` by
Jensen's inequality, so removing inflation uncertainty overstates how much
inflation erodes -- but only slightly.  The paper prices exactly this
comparison: counterfactual 4 ("higher for sure", deterministic inflation at
the mean) gives 3.8% of GDP against 3.7% for the full stochastic model.

It is not cheap on the maturity side.  See :mod:`debtmodel.maturity`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .curves import TermStructure
from .maturity import DebtProfile

__all__ = ["DebasementModel", "RepressionResult"]


@dataclass(frozen=True)
class DebasementModel:
    """Discrete-time, fixed-parameter valuation of nominal government debt.

    Parameters
    ----------
    real_rates:
        Term structure of annualised real risk-free rates, ``r_t``.  Held
        fixed across counterfactuals: the paper's experiments shift the
        risk-adjusted *inflation* distribution while leaving real rates alone,
        so that the change in ``W`` isolates debasement.
    inflation:
        Term structure of average annual inflation to each horizon.  This is
        the harmonic mean the paper reports in table 4 -- harmonic because
        Proposition 1 makes the expectation of *inverse* inflation the
        relevant moment, so a harmonic mean is the deterministic rate that
        reproduces the correct weight.
    """

    real_rates: TermStructure
    inflation: TermStructure

    # -- the weights of Proposition 1 ------------------------------------

    def weights(self, horizons: np.ndarray) -> np.ndarray:
        """``omega_t`` of equation (8) at each horizon.

        With deterministic inflation this is just the nominal discount factor,
        which is the point: under fixed parameters the burden ``W`` is the
        market value of the debt, and a counterfactual is a repricing.
        """
        t = np.asarray(horizons, dtype=float)
        return (1.0 + self.real_rates.at(t)) ** -t * (1.0 + self.inflation.at(t)) ** -t

    def nominal_rates(self, horizons: np.ndarray) -> np.ndarray:
        """The implied annualised nominal yield at each horizon (Fisher)."""
        t = np.asarray(horizons, dtype=float)
        return (1.0 + self.real_rates.at(t)) * (1.0 + self.inflation.at(t)) - 1.0

    # -- valuation --------------------------------------------------------

    def value(self, profile: DebtProfile) -> float:
        """The debt burden ``W`` for a given maturity profile, equation (7)."""
        return float((self.weights(profile.horizons) * profile.payments).sum())

    def duration(self, profile: DebtProfile) -> float:
        """Macaulay duration in years: present-value-weighted average maturity.

        Shorter than :attr:`DebtProfile.average_maturity`, and the quantity
        that actually governs the first-order sensitivity of ``W`` to a
        parallel inflation shift.
        """
        w = self.weights(profile.horizons) * profile.payments
        return float((profile.horizons * w).sum() / w.sum())

    # -- counterfactuals ---------------------------------------------------

    def with_inflation(self, inflation: TermStructure) -> "DebasementModel":
        """The same real rates, a different inflation term structure."""
        return replace(self, inflation=inflation)

    def shift_inflation(self, delta: TermStructure | float) -> "DebasementModel":
        """The same real rates, inflation raised by ``delta``."""
        return replace(self, inflation=self.inflation.shifted(delta))

    def debasement(
        self, profile: DebtProfile, counterfactual: "DebasementModel"
    ) -> float:
        """Fall in the real present value of ``profile`` moving to ``counterfactual``.

        Positive means the government gains: the debt is worth less.
        """
        return self.value(profile) - counterfactual.value(profile)

    # -- financial repression (section 8) ----------------------------------

    def repression(
        self,
        profile: DebtProfile,
        counterfactual: "DebasementModel",
        years: float,
    ) -> "RepressionResult":
        """Decompose the joint effect of ``years`` of extreme financial repression.

        Extreme repression pays bondholders in required reserves bearing zero
        interest which they must hold for ``years`` periods.  Proposition 3
        shows this is equivalent to deferring every payment by that many years
        at no interest, so ``omega_t`` is replaced by ``omega_{t+N}``.

        The split matches table 6: repression alone at unchanged inflation,
        then the inflation shift applied inside the repressed economy.
        """
        deferred = profile.deferred(years)
        base = self.value(profile)
        repressed = self.value(deferred)
        both = counterfactual.value(deferred)
        return RepressionResult(
            years=float(years),
            from_repression=base - repressed,
            from_inflation=repressed - both,
            total=base - both,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_nominal_curve(
        cls, nominal: TermStructure, inflation: TermStructure
    ) -> "DebasementModel":
        """Build a model from an observed nominal yield curve plus inflation.

        This is the calibration that keeps the model honest.  The nominal
        curve is observable and is what actually prices the debt, so the
        baseline ``W`` reproduces the debt's market value by construction;
        real rates are then whatever the Fisher equation implies given the
        inflation term structure, and it is *those* that are held fixed when
        inflation shifts.
        """
        tenors = np.union1d(nominal.tenors, inflation.tenors)
        real = (1.0 + nominal.at(tenors)) / (1.0 + inflation.at(tenors)) - 1.0
        return cls(
            real_rates=TermStructure(tenors, real, name="real"),
            inflation=inflation,
        )


@dataclass(frozen=True)
class RepressionResult:
    """The table 6 decomposition, in currency units."""

    years: float
    from_repression: float
    from_inflation: float
    total: float
