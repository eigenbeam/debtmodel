"""The same model, two dates: end-2012 (the paper's snapshot) and September 2026.

    uv run python examples/compare_2012_2026.py
"""

from __future__ import annotations

from debtmodel import calibration as c12
from debtmodel import calibration_2026 as c26


def main() -> None:
    m12, d12, g12 = c12.baseline_model(), c12.private_holdings(), c12.GDP_2012
    m26, d26, g26 = c26.baseline_model_2026(), c26.private_holdings_2026(), c26.GDP_2026

    rows = [
        ("nominal GDP, $tn", g12 / 1000, g26 / 1000, "{:.1f}"),
        ("nominal debt priced, $tn", m12.value(d12) / 1000, m26.value(d26) / 1000, "{:.1f}"),
        ("  as a share of GDP", m12.value(d12) / g12, m26.value(d26) / g26, "{:.1%}"),
        ("average maturity, years", d12.average_maturity, d26.average_maturity, "{:.2f}"),
        ("Macaulay duration, years", m12.duration(d12), m26.duration(d26), "{:.2f}"),
    ]
    print("=" * 62)
    print(f"{'':32}{'end-2012':>13}{'Sep-2026':>13}")
    print("=" * 62)
    for label, a, b, fmt in rows:
        print(f"{label:32}{fmt.format(a):>13}{fmt.format(b):>13}")

    print()
    print("Term structure")
    for t in (1, 5, 10, 30):
        print(
            f"  {t:2d}y  nominal {m12.nominal_rates(t):+6.2%} -> {m26.nominal_rates(t):+6.2%}"
            f"   real {m12.real_rates.at(t):+6.2%} -> {m26.real_rates.at(t):+6.2%}"
            f"   inflation {m12.inflation.at(t):5.2%} -> {m26.inflation.at(t):5.2%}"
        )

    print()
    print("=" * 62)
    print("Debasement: fall in the real PV of debt, % of GDP")
    print("=" * 62)
    print(f"{'':32}{'end-2012':>13}{'Sep-2026':>13}")
    cf12 = c12.counterfactual_model(1)
    cf26 = c26.counterfactual_cf1_2026()
    print(
        f"{'HRR counterfactual 1':32}"
        f"{m12.debasement(d12, cf12) / g12:>13.2%}"
        f"{m26.debasement(d26, cf26) / g26:>13.2%}"
    )
    for bp in (100, 200, 500):
        a = m12.debasement(d12, m12.shift_inflation(bp / 1e4)) / g12
        b = m26.debasement(d26, m26.shift_inflation(bp / 1e4)) / g26
        print(f"{f'parallel +{bp / 100:.0f}pp':32}{a:>13.2%}{b:>13.2%}")

    print()
    print("Extreme financial repression, total effect (% of GDP)")
    for years in (1, 5, 10):
        a = m12.repression(d12, cf12, years).total / g12
        b = m26.repression(d26, cf26, years).total / g26
        print(f"{f'  {years} years':32}{a:>13.2%}{b:>13.2%}")

    print()
    print("=" * 62)
    a = m12.debasement(d12, m12.shift_inflation(0.01)) / g12
    b = m26.debasement(d26, m26.shift_inflation(0.01)) / g26
    print(
        f"Erosion per point of inflation rose {b / a:.2f}x, while the debt rose "
        f"{(m26.value(d26) / g26) / (m12.value(d12) / g12):.2f}x\n"
        f"and duration barely moved ({m12.duration(d12):.2f}y -> {m26.duration(d26):.2f}y).\n"
        f"The change is the size of the debt, not its structure."
    )


if __name__ == "__main__":
    main()
