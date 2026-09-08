"""Term structures: annual grids of rates, interpolated from sparse quotes.

Every curve in the model is a function of an integer horizon ``t`` in years.
Quotes are given at the handful of maturities that are actually observable
(the constant-maturity Treasury tenors, the horizons in the paper's tables)
and filled in between by linear interpolation in the rate, flat beyond the
ends.  Linear-in-rate is what the paper's own inputs are reported as, and at
annual resolution the choice of interpolator is immaterial next to the
uncertainty in the underlying quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TermStructure"]


@dataclass(frozen=True)
class TermStructure:
    """An annualised rate quoted at a set of maturities, in decimal (0.025 = 2.5%).

    Parameters
    ----------
    tenors:
        Maturities in years at which the rate is quoted.  Must be increasing.
    rates:
        The annualised rate at each tenor, as a decimal.
    name:
        Label used in reprs and error messages.
    """

    tenors: np.ndarray
    rates: np.ndarray
    name: str = "rate"

    def __post_init__(self) -> None:
        tenors = np.asarray(self.tenors, dtype=float)
        rates = np.asarray(self.rates, dtype=float)
        if tenors.shape != rates.shape:
            raise ValueError(
                f"{self.name}: tenors and rates must have the same shape, "
                f"got {tenors.shape} and {rates.shape}"
            )
        if tenors.ndim != 1 or tenors.size == 0:
            raise ValueError(f"{self.name}: tenors must be a non-empty 1-D array")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError(f"{self.name}: tenors must be strictly increasing")
        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "rates", rates)

    def at(self, t: np.ndarray | float) -> np.ndarray:
        """The rate at horizon(s) ``t``, linearly interpolated, flat outside."""
        return np.interp(np.asarray(t, dtype=float), self.tenors, self.rates)

    def shifted(self, delta: "TermStructure | float") -> "TermStructure":
        """This curve plus ``delta``, evaluated on the union of both tenor sets.

        A scalar shifts the curve in parallel.  Another ``TermStructure``
        shifts it by a term structure, which is how the paper's counterfactuals
        actually move: inflation rises more at long horizons than short ones.
        """
        if isinstance(delta, TermStructure):
            tenors = np.union1d(self.tenors, delta.tenors)
            rates = self.at(tenors) + delta.at(tenors)
        else:
            tenors = self.tenors
            rates = self.rates + float(delta)
        return TermStructure(tenors, rates, name=f"{self.name}+shift")

    @classmethod
    def flat(cls, rate: float, name: str = "rate") -> "TermStructure":
        """A constant rate at every horizon."""
        return cls(np.array([1.0]), np.array([float(rate)]), name=name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        pairs = ", ".join(f"{t:g}y={r:.2%}" for t, r in zip(self.tenors, self.rates))
        return f"TermStructure({self.name}: {pairs})"
