"""Project every scenario and print the comparison tables.

    uv run python examples/run_scenarios.py
"""

from __future__ import annotations

from debtmodel.scenarios import GROUPS, required_primary_balance, run_all


def main() -> None:
    r = run_all()
    base = r["baseline"]

    print("=" * 78)
    print("Debt held by the public, % of GDP")
    print("=" * 78)
    print(f"  {'':44}{'2036':>8}{'2050':>8}{'vs base':>10}")
    for group, names in GROUPS.items():
        print(f"\n  {group}")
        for n in names:
            p = r[n]
            gap = (p.debt_gdp[-1] - base.debt_gdp[-1]) * 100
            flag = "  stable" if p.stabilises() else ""
            print(f"    {p.label:42}{p.debt_gdp[10]:>8.0%}{p.debt_gdp[-1]:>8.0%}"
                  f"{gap:>+9.0f}pp{flag}")

    print()
    print("=" * 78)
    print("What stabilising requires: constant primary balance holding debt flat")
    print("=" * 78)
    rows = [
        ("baseline conditions", {}),
        ("if real rates are 100bp higher", dict(real_short_rate=0.022)),
        ("if growth is 1.0% not 1.8%", dict(real_growth=0.010)),
        ("if growth is 3.1%", dict(real_growth=0.031)),
        ("with 175bp of financial repression", dict(repression_bp=175.0)),
        ("with inflation at 4.3% not 2.3%", dict(inflation=0.043)),
    ]
    for label, kw in rows:
        need = required_primary_balance(**kw)
        print(f"  {label:38}{need:>+8.1%}   adjustment needed {need + 0.025:>+6.1%}")
    print(f"\n  Currently running a primary deficit of 2.5% of GDP.")

    print()
    print("=" * 78)
    print("Treasury Twist as a bet")
    print("=" * 78)
    t = (r["twist"].terminal_debt - base.terminal_debt) * 100
    s = (r["spike_bill_funded"].terminal_debt
         - r["spike_long_funded"].terminal_debt) * 100
    print(f"  if rates follow the forecast   {t:+.1f}pp of GDP by 2050")
    print(f"  if a rate spike surprises      {s:+.1f}pp of GDP by 2050")
    print(f"  breakeven probability of a spike: {abs(t) / (abs(t) + abs(s)):.0%}")


if __name__ == "__main__":
    main()
