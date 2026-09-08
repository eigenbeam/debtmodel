"""The maturity profile of nominal payments, ``B_0^t`` in Proposition 1.

``B_0^t`` is the total nominal dollar amount -- principal *and* coupon -- that
the government has promised to pay to a given set of holders at date ``t``.
It is data, fixed in the past, and it is the single most important input to
the model: Hilscher, Raviv & Reis's table 5 shows that replacing the true
profile with a single bond at the average maturity cuts the estimated
debasement from 3.7% of GDP to 1.1%.  So this class deliberately refuses to
collapse the profile to a summary statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DebtProfile"]


@dataclass(frozen=True)
class DebtProfile:
    """Nominal payments due at each future year.

    Parameters
    ----------
    payments:
        ``payments[i]`` is the nominal amount due at horizon ``horizons[i]``,
        in whatever currency unit the caller chooses (the shipped calibration
        uses billions of USD).
    horizons:
        Payment dates in years.  Defaults to ``1, 2, ..., len(payments)``.
        Horizon 0 is permitted and represents payments due immediately, which
        no amount of future inflation can erode.
    label:
        Who holds this debt, e.g. ``"privately held"``.
    """

    payments: np.ndarray
    horizons: np.ndarray | None = None
    label: str = "debt"

    def __post_init__(self) -> None:
        payments = np.asarray(self.payments, dtype=float)
        if payments.ndim != 1 or payments.size == 0:
            raise ValueError(f"{self.label}: payments must be a non-empty 1-D array")
        if np.any(payments < 0):
            raise ValueError(f"{self.label}: payments must be non-negative")
        if self.horizons is None:
            horizons = np.arange(1.0, payments.size + 1.0)
        else:
            horizons = np.asarray(self.horizons, dtype=float)
            if horizons.shape != payments.shape:
                raise ValueError(
                    f"{self.label}: horizons and payments must have the same shape, "
                    f"got {horizons.shape} and {payments.shape}"
                )
            if np.any(np.diff(horizons) <= 0):
                raise ValueError(f"{self.label}: horizons must be strictly increasing")
            if horizons[0] < 0:
                raise ValueError(f"{self.label}: horizons must be non-negative")
        object.__setattr__(self, "payments", payments)
        object.__setattr__(self, "horizons", horizons)

    @property
    def face_value(self) -> float:
        """Total undiscounted nominal payments owed."""
        return float(self.payments.sum())

    @property
    def average_maturity(self) -> float:
        """Payment-weighted average maturity, in years.

        This is the ``4.8 years`` reported in section 3.2 of the paper -- an
        undiscounted, payment-weighted mean, not a Macaulay duration.
        """
        return float((self.horizons * self.payments).sum() / self.payments.sum())

    def truncated(self, max_maturity: float) -> "DebtProfile":
        """Only the payments due at or before ``max_maturity``.

        Used for the table 5 counterfactuals, which ask how much of the
        debasement comes from debt maturing within 1 year, or within 4.5 years.
        """
        keep = self.horizons <= max_maturity
        if not keep.any():
            raise ValueError(
                f"{self.label}: no payments at or before {max_maturity} years"
            )
        return DebtProfile(
            self.payments[keep],
            self.horizons[keep],
            label=f"{self.label} (<={max_maturity:g}y)",
        )

    def scaled(self, factor: float) -> "DebtProfile":
        """The same maturity shape, scaled by ``factor``."""
        return DebtProfile(self.payments * factor, self.horizons, label=self.label)

    def deferred(self, years: float) -> "DebtProfile":
        """Every payment pushed back by ``years``, at no interest.

        This is the exact form that extreme financial repression takes in the
        paper: section 8 shows that paying bondholders in zero-interest
        required reserves they must hold for N periods is equivalent to
        shifting the whole maturity structure out by N years.
        """
        if years < 0:
            raise ValueError(f"{self.label}: cannot defer by {years} years")
        return DebtProfile(
            self.payments,
            self.horizons + float(years),
            label=f"{self.label} (deferred {years:g}y)",
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DebtProfile({self.label}: face={self.face_value:,.0f}, "
            f"avg maturity={self.average_maturity:.2f}y, "
            f"{self.horizons[0]:g}-{self.horizons[-1]:g}y)"
        )
