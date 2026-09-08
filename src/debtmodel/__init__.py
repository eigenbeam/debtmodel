"""A discrete-time model of the US public debt burden and its erosion by inflation.

Implements Proposition 1 of Hilscher, Raviv & Reis, "Inflating Away the Public
Debt? An Empirical Assessment" (NBER WP 20339, 2014): the real value of the
outstanding nominal debt is a weighted average of the nominal payments due at
each future date, with weights given by the expected inverse of compounded
risk-adjusted inflation.

This version fixes the parameters -- inflation is deterministic given a term
structure of average annual rates -- which the paper's own table 3 suggests
costs little (3.8% vs 3.7% of GDP), while keeping the full maturity profile,
which it shows costs a great deal to drop (1.1% vs 3.7%).
"""

from .burden import DebasementModel, RepressionResult
from .curves import TermStructure
from .maturity import DebtProfile

__all__ = ["DebasementModel", "DebtProfile", "RepressionResult", "TermStructure"]
__version__ = "0.1.0"
