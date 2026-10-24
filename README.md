# debtmodel

A discrete-time model of the US public debt burden and how much of it inflation
can erode, implementing Hilscher, Raviv & Reis, *"Inflating Away the Public
Debt? An Empirical Assessment"* (NBER WP 20339, 2014).

## The model

Proposition 1 of the paper: the real market value of outstanding nominal debt is
a weighted sum of the nominal payments due at every future date,

```
W = Σ_t ω_t B_0^t          ω_t = R_t⁻¹ · E^Q[1 / π_{0,t}]
```

where `B_0^t` is the nominal payment (principal and coupon) due at `t`, `R_t` is
the gross real risk-free return to `t`, and `π_{0,t} = P_t/P_0` is gross
cumulative inflation under the risk-adjusted measure. Debasement is the fall in
`W` when the inflation distribution shifts while real rates are held fixed.

This is the **fixed-parameter** version: inflation is deterministic given a term
structure of average annual rates, so `E^Q[1/π_{0,t}]` collapses to
`(1 + π̄_t)^-t` and `ω_t` is just a nominal discount factor.

### Two simplifications, priced

The paper lets you cost both of the obvious shortcuts, and they are not
comparable:

| Shortcut | Paper's own estimate | Verdict |
|---|---|---|
| Deterministic inflation at the mean | 3.8% vs 3.7% of GDP (table 3, rows 4 vs 1) | Cheap — taken here |
| One bond at the average maturity | 1.1% vs 3.7% of GDP (table 5) | Ruinous — refused here |

So `DebtProfile` keeps the full payment schedule and `TermStructure` keeps a
full term structure of inflation. A scalar `π` and a 4.8-year average maturity
would reproduce exactly the back-of-envelope the paper's introduction sets out
to demolish.

## Parameter sets

Two calibrations, side by side. `calibration.py` (end-2012) is the paper's own
snapshot and exists to show the model reproduces published results;
`calibration_2026.py` (September 2026) is the current US position. The 2012 set
is frozen — the 2026 set adds to it and changes nothing in it.

## Usage

```bash
uv sync
uv run python examples/replicate_hrr.py       # reproduce the paper's tables (2012)
uv run python examples/compare_2012_2026.py   # the two dates side by side
uv run --extra dev pytest                     # 45 tests against published numbers
```

```python
from debtmodel.calibration import baseline_model, counterfactual_model, private_holdings, GDP_2012

model, debt = baseline_model(), private_holdings()
model.value(debt) / GDP_2012                                  # 0.512 — the burden today
model.debasement(debt, counterfactual_model(1)) / GDP_2012    # 0.037 — permanently higher inflation
model.debasement(debt, model.shift_inflation(0.01)) / GDP_2012  # 0.019 — a flat +1pp
model.repression(debt, counterfactual_model(1), years=10)     # section 8
```

```python
from debtmodel.calibration_2026 import baseline_model_2026, private_holdings_2026, counterfactual_cf1_2026, GDP_2026

model, debt = baseline_model_2026(), private_holdings_2026()
model.value(debt) / GDP_2026                                     # 0.728
model.debasement(debt, counterfactual_cf1_2026()) / GDP_2026     # 0.052
model.debasement(debt, model.shift_inflation(0.01)) / GDP_2026   # 0.026
```

## Replication

| Target | Model | Paper |
|---|---|---|
| Market value of privately-held nominals | $8.29tn (51.2% GDP) | $8.3tn (51.2%) |
| Average maturity | 4.85y | 4.8y |
| CF1, permanently higher inflation | 3.7% GDP | 3.7% |
| CF2, right tail only | 4.5% | 4.4% |
| CF4, higher for sure | 3.8% | 3.8% |
| CF5, partially anticipated | 1.4% | 1.3% |
| CF7, gradual increase | 2.1% | 2.3% |
| Restricted to ≤1y maturities | 0.1% | 0.1% |
| Restricted to ≤4.5y maturities | 1.2% | 1.1% |
| Repression 10y (repression / inflation / total) | 10.9 / 10.3 / 21.2% | 12.2 / 10.7 / 22.9% |

Three of these are calibration targets (market value, average maturity, CF1).
The rest are out of sample, and the repression block is entirely so — nothing in
the calibration touches section 8.

### Two independent consistency checks

Neither is targeted, and both could have failed.

**The implied real curve.** Deflating the observed end-2012 nominal Treasury
curve by the paper's table 4 inflation gives real rates of −1.6% at 5y, −0.7% at
10y and −0.1% at 30y. Actual end-2012 TIPS yields were about −1.4%, −0.7% and
+0.4%. The paper's option-implied inflation and the Treasury curve agree with the
TIPS market they were never fitted to.

**Repression.** Deferring every payment by N years at zero interest — which
Proposition 3 shows is what extreme financial repression amounts to — reproduces
table 6 to within 1.7pp of GDP at a 22.9% effect.

## Calibration and its weak point

| Input | Source | Confidence |
|---|---|---|
| Nominal yield curve | Treasury CMT, 31 Dec 2012 | Public data |
| Inflation term structures | Paper's table 4 | Published |
| Maturity profile `B_0^t` | **Reconstructed** | **Weak** |

The maturity profile is the weak link and it is also the input that matters
most. The paper builds it from security-level CRSP data that is not reproduced
in the text, so `calibration._PROFILE_SHAPE` is a shape read off figure 2, with
the long end damped and the whole schedule rescaled to hit the paper's published
moments. It reproduces those moments and predicts the untargeted results well,
but it is a reconstruction, not data. Replace it with a real payment schedule
before leaning on any number here.

Two smaller conventions worth knowing. Annual buckets sit at their **midpoints**
(`t = 0.5, 1.5, …`), which is not cosmetic: the first bucket holds 35% of the
value, and placing it at `t=1` doubles its contribution to debasement. And the
inflation rates in table 4 are **harmonic** means, because Proposition 1 makes
`E[1/π]` the relevant moment — a harmonic mean is the deterministic rate that
reproduces the correct weight.

## Known limitation

The fixed-parameter model cannot see inflation *variance*. Counterfactuals 1 and
3 differ almost only in variability; the paper separates them (3.7% vs 3.4% of
GDP) because `E[1/π] > 1/E[π]` by Jensen, so uncertainty raises the value of
nominal payments and blunts debasement. This model collapses them to 3.7% and
3.7%. That gap — 0.3% of GDP — is the entire price of dropping the stochastic
machinery, and it is the natural next extension: replace the deterministic
`ω_t` with an integral against a risk-adjusted density recovered from inflation
option prices (paper §4–5).

The extension point is `DebasementModel.weights`; nothing else in the model
needs to know whether inflation is deterministic.

## September 2026

|  | end-2012 | Sep-2026 |
|---|---|---|
| Nominal GDP | $16.2tn | $32.5tn |
| Nominal debt priced by the model | $8.3tn | $23.7tn |
| — as a share of GDP | 51.2% | **72.8%** |
| Macaulay duration | 3.93y | 3.85y |
| 10y real rate | −0.72% | **+2.43%** |
| 30y inflation compensation | 3.00% | 2.22% |
| HRR counterfactual 1 | 3.7% of GDP | **5.2%** |
| A permanent +1pp at every horizon | 1.9% | 2.6% |
| Repression, 10 years, total | 21.3% | 39.1% |

**The finding.** Erosion per point of inflation rose 1.39×; the debt rose 1.42×;
duration moved 3.93 → 3.85 years. Inflation is a more powerful fiscal
instrument in 2026 than in 2012 almost exactly in proportion to there being more
debt, and for no other reason. That is worth stating because the intuitive story
— that a decade of heavy long-bond issuance lengthened the debt and made it more
inflatable — is not what happened. Par-weighted maturity did lengthen (Treasury's
WAM is ~71 months), but the *duration that matters for debasement* is flat,
because the front end grew just as fast and higher discount rates pull weight
back toward it.

Two offsetting changes sit underneath. Real rates went from negative across the
curve to +2.4% at ten years, which shortens effective duration and cuts the
weight on distant payments. Inflation compensation flattened and fell at the long
end (3.0% → 2.2% at 30 years). Neither rescues the conclusion: the paper's own
counterfactual applied to today's debt erodes 5.2% of GDP against a 73%-of-GDP
stock, and a permanent extra point of inflation still buys 2.6%.

### Sources and provenance, 2026

| Input | Source | Confidence |
|---|---|---|
| Maturity profile `B_0^t` | MSPD table 3, security level, 31 Aug 2026, netted CUSIP-by-CUSIP against NY Fed SOMA holdings, 2 Sep 2026 | **Strong** |
| Nominal yield curve | Treasury CMT, 4 Sep 2026 | Public data |
| Real yield curve | Treasury real CMT (TIPS), 4 Sep 2026 | Public data |
| Inflation term structure | Breakeven, `(1+nominal)/(1+real) − 1`, pinned to the quoted 2.31% 1y breakeven at the short end | Public data |
| Counterfactual shift | **Transplanted** from the paper's table 4 | **Weak** |

The 2012 set's weakest input is the 2026 set's strongest. The maturity profile is
no longer read off a chart: `scripts/build_2026_profile.py` pulls every
marketable bill, note and bond from the Treasury's MSPD API, expands each into
its actual coupon and principal payments, and subtracts the Fed's holdings of
each CUSIP. Category totals reconcile exactly to the MSPD summary ($7.248tn
bills, $16.218tn notes, $5.525tn bonds, $2.153tn TIPS, $0.680tn FRNs), and the
par-weighted maturity implied by the parsed list is 71.3 months against
Treasury's separately published ~70 — an external check that the parse is right.

TIPS ($2.15tn) are excluded: their payments are indexed, and equation (5) of the
paper shows debasement turns on nominal debt alone. FRNs are charged to their
next quarterly reset rather than their maturity, since a floating coupon is not
fixed in nominal terms beyond one period.

Two caveats. The measure is *held outside the Federal Reserve*, which is slightly
wider than the paper's "privately held" — it still includes state and local
government holdings, which the paper further removed (≈7.5% of the total in
2012). There is no comparably clean current source for that split, so it is left
in and flagged rather than guessed at.

And the counterfactual is transplanted. The paper sized its shift against
option-implied inflation densities recovered from caps and floors; this module
has breakeven inflation, which is a risk-adjusted *mean* but not a distribution.
`HRR_CF1_SHIFT` therefore reuses the paper's 2012 increments (+1.7pp at 1y rising
to +2.6pp at 30y) on the 2026 base. It answers "what would the paper's
experiment do to today's debt?" — not "what do 2026 markets price as possible?"
Answering the second needs the option data of the paper's §4–5.

Note also that the two `real_rates` curves are epistemically different. In 2012
the real curve was *inferred* by deflating the nominal curve with the paper's
option-implied inflation, and matching the observed TIPS curve was a genuine
out-of-sample check. In 2026 both curves are observed and breakevens are defined
as their ratio, so the corresponding test is a round-trip and is labelled as one.

## Structure

```
src/debtmodel/
  curves.py             TermStructure — annual rate grids, interpolation, shifts
  maturity.py           DebtProfile — B_0^t, truncation, deferral (repression)
  burden.py             DebasementModel — ω_t, valuation, counterfactuals, repression
  calibration.py        US 2012 inputs and the paper's published targets
  calibration_2026.py   US September 2026 inputs
scripts/
  build_2026_profile.py Regenerates the 2026 payment profile from Treasury/NY Fed APIs
```

## The result

Inflation does not pay off the US public debt, at either date. Erosion is
roughly `duration × value`: a 3.9-year duration on debt worth 51% of GDP gave
about 1.9% of GDP per point of inflation in 2012, and a 3.85-year duration on
debt worth 73% of GDP gives about 2.6% today. Fourteen years, a debt that grew
by half relative to GDP, and a complete reversal in real rates moved that number
by 39%.

Getting to a large number requires either inflation far outside anything markets
price as possible, or a long maturity structure the Treasury does not have. That
is why financial repression — an ex post extension of maturity — remains the one
lever that changes the answer by an order of magnitude, and why it grew more
potent than inflation did: ten years of it now reaches 39% of GDP, against 21%
in 2012. The debt is bigger and nominal rates are no longer at zero, so forcing
holders into non-interest-bearing claims costs them far more than it used to.
