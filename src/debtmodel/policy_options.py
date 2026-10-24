"""What the levers mean in practice.

The projections deal in abstractions -- "a 2.5-point swing in the primary
balance", "175 basis points of repression". This module attaches those to
things a legislature could actually vote for, sized against CBO's own scores
so the arithmetic of what it takes is concrete rather than rhetorical.

The central fact it exists to make visible: the largest single option CBO
scores is worth about 0.9% of GDP a year. Stabilising needs roughly two and a
half of those, simultaneously and permanently.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Option", "REVENUE", "SPENDING", "REPRESSION", "as_share_of_gdp",
           "CUMULATIVE_GDP_10Y", "largest_single_option", "options_needed"]

#: Cumulative nominal GDP over CBO's 2025-2034 scoring window, $bn, built from
#: $31.2tn in 2025 growing at the baseline's 4.1% nominal rate. CBO scores
#: options as ten-year totals, so this is the right denominator for converting
#: them into an annual share of GDP.
CUMULATIVE_GDP_10Y = 372_000.0


def as_share_of_gdp(ten_year_total_bn: float) -> float:
    """Convert a CBO ten-year score into an average annual share of GDP."""
    return ten_year_total_bn / CUMULATIVE_GDP_10Y


@dataclass(frozen=True)
class Option:
    """One scored policy change."""

    label: str
    ten_year_bn: float
    note: str = ""

    @property
    def share_of_gdp(self) -> float:
        return as_share_of_gdp(self.ten_year_bn)

    @property
    def annual_bn(self) -> float:
        return self.ten_year_bn / 10.0


#: Revenue options, from CBO's "Options for Reducing the Deficit: 2025 to 2034".
REVENUE = [
    Option("Eliminate all itemised deductions", 3_400,
           "The largest single option CBO scores. Removes the mortgage "
           "interest, charitable and state-and-local deductions together."),
    Option("5% value-added tax, broad base", 3_400,
           "A tax the US does not currently levy and most peer economies do. "
           "The range depends on whether food, health and housing are exempt."),
    Option("5% value-added tax, narrow base", 2_200,
           "The same rate with the usual exemptions carved out."),
    Option("New 2% payroll tax", 2_500,
           "Applied to all earnings; roughly a fifth again on the existing "
           "Social Security levy."),
    Option("Surtax on adjusted gross income", 1_400,
           "Top of CBO's scored range."),
    Option("New 1% payroll tax", 1_300),
]

#: Spending options.
SPENDING = [
    Option("Cap Medicaid spending per enrollee", 893,
           "Converts the open-ended federal match into a per-person cap."),
    Option("Reduce Medicare Advantage risk-adjustment payments", 1_000,
           "Top of a wide scored range; the low end is $124bn."),
    Option("Reduce the defence budget", 959,
           "Roughly a tenth of defence spending, sustained."),
    Option("Uniform Social Security benefit at 125% of poverty", 607,
           "Flattens the benefit formula; a large change to the programme's "
           "structure rather than a trim."),
    Option("Cap total federal Medicaid funding", 459),
]

#: Repression is not legislated as a budget line, so it has no CBO score.
#: These are the instruments that produce it, with the mechanism in each case.
REPRESSION = [
    ("Central bank holdings",
     "The most direct instrument, and the one already in use. The Federal "
     "Reserve holds 14% of publicly held US debt; the Bank of Japan holds 42% "
     "of JGBs and T-bills. Moving toward the Japanese share would mean a "
     "quantitative easing programme of several trillion dollars, held "
     "indefinitely rather than unwound."),
    ("Bank liquidity and capital rules",
     "Treasuries already carry a zero risk weight and count as the highest "
     "grade of liquid asset banks must hold. Tightening liquidity coverage "
     "ratios, or widening which institutions they bind, mechanically "
     "manufactures captive demand without any appearance of a levy."),
    ("Pension and insurance solvency regulation",
     "Rules that treat government bonds as the risk-free matching asset push "
     "long-duration institutional money into the long end. This is the "
     "quietest instrument and, in the European experience, among the "
     "most powerful."),
    ("Reserve requirements paid below market",
     "The mechanism modelled in the source paper: bondholders are paid in "
     "reserves they must hold, at a rate below what they could earn "
     "elsewhere. Equivalent to an ex post extension of maturity."),
    ("Restrictions on the alternatives",
     "Limits on foreign asset holdings, on capital outflows, or on the "
     "instruments funds may hold. The postwar system rested on these; they "
     "are the hardest to reintroduce in an economy with open capital "
     "markets, and the most damaging if tried."),
]


def largest_single_option() -> Option:
    """The biggest thing on CBO's menu."""
    return max(REVENUE + SPENDING, key=lambda o: o.ten_year_bn)


def options_needed(swing: float) -> float:
    """How many of the largest scored option a given swing amounts to."""
    return swing / largest_single_option().share_of_gdp
