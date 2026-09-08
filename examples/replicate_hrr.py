"""Reproduce the main tables of Hilscher, Raviv & Reis (2014) side by side.

    uv run python examples/replicate_hrr.py
"""

from __future__ import annotations

import numpy as np

from debtmodel.calibration import (
    GDP_2012,
    INFLATION_BASELINE,
    INFLATION_COUNTERFACTUALS,
    PAPER_TABLE_3_PRIVATE,
    baseline_model,
    counterfactual_model,
    private_holdings,
)


def pct(x: float) -> str:
    return f"{x:5.1%}"


def main() -> None:
    model = baseline_model()
    debt = private_holdings()

    print("=" * 66)
    print("The debt, end-2012            model      paper")
    print("=" * 66)
    print(f"  market value            ${model.value(debt) / 1000:6.2f}tn   $8.3tn")
    print(f"  as a share of GDP        {model.value(debt) / GDP_2012:6.1%}    51.2%")
    print(f"  average maturity         {debt.average_maturity:6.2f}y     4.8y")
    print(f"  Macaulay duration        {model.duration(debt):6.2f}y        -")

    print()
    print("Implied real term structure (not targeted; compare end-2012 TIPS)")
    for t, tips in [(5, "-1.4%"), (10, "-0.7%"), (30, "+0.4%")]:
        print(f"  {t:2d}y  {model.real_rates.at(t):+6.2%}     TIPS {tips}")

    print()
    print("=" * 66)
    print("Table 3: fall in the real PV of debt, % of GDP (privately held)")
    print("=" * 66)
    print(f"  {'counterfactual':<28}{'model':>8}{'paper':>8}   diff")
    for n, infl in INFLATION_COUNTERFACTUALS.items():
        got = model.debasement(debt, counterfactual_model(n)) / GDP_2012
        want = PAPER_TABLE_3_PRIVATE[n]
        note = "  <- variance effect, unseen here" if n == 3 else ""
        print(f"  {n}. {infl.name:<25}{got:8.1%}{want:8.1%}{got - want:+7.1%}{note}")

    print()
    print("=" * 66)
    print("Table 5: the same shift, restricted to short maturities")
    print("=" * 66)
    cf1 = counterfactual_model(1)
    for cap, want in [(1.0, 0.001), (4.5, 0.011), (30.0, 0.037)]:
        sub = debt.truncated(cap)
        got = model.debasement(sub, cf1) / GDP_2012
        share = model.value(sub) / model.value(debt)
        print(
            f"  maturities <= {cap:4.1f}y   holds {share:5.1%} of the value   "
            f"{got:6.1%}   paper {want:.1%}"
        )

    print()
    print("=" * 66)
    print("Table 6: extreme financial repression, % of GDP")
    print("=" * 66)
    print(f"  {'years':>7}{'repression':>13}{'inflation':>11}{'total':>8}    paper")
    for years, want in [(1, "0.7 / 4.6 / 5.3"), (5, "4.8 / 8.1 / 12.9"),
                        (10, "12.2 / 10.7 / 22.9")]:
        r = model.repression(debt, cf1, years)
        print(
            f"  {years:>7}{r.from_repression / GDP_2012:>13.1%}"
            f"{r.from_inflation / GDP_2012:>11.1%}{r.total / GDP_2012:>8.1%}"
            f"    {want}"
        )

    print()
    print("=" * 66)
    print("Sensitivity: parallel shift in inflation at every horizon")
    print("=" * 66)
    for bp in (50, 100, 200, 500, 1000):
        shifted = model.shift_inflation(bp / 10_000)
        fall = model.debasement(debt, shifted) / GDP_2012
        print(f"  +{bp / 100:5.1f}pp   ->  {fall:6.2%} of GDP")
    print()
    print("  Note the near-linearity: erosion is roughly duration x value x shift,")
    print(f"  i.e. {model.duration(debt):.1f}y x {model.value(debt) / GDP_2012:.0%}"
          f" ~ {model.duration(debt) * model.value(debt) / GDP_2012:.1f}% of GDP per pp.")
    print("  There is no point on this curve where inflation pays off the debt.")


if __name__ == "__main__":
    main()
