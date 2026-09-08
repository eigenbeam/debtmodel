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

## Usage

```bash
uv sync
uv run python examples/replicate_hrr.py    # reproduce the paper's tables
uv run --extra dev pytest                  # 28 tests, all against published numbers
```

```python
from debtmodel.calibration import baseline_model, counterfactual_model, private_holdings, GDP_2012

model, debt = baseline_model(), private_holdings()
model.value(debt) / GDP_2012                                  # 0.512 — the burden today
model.debasement(debt, counterfactual_model(1)) / GDP_2012    # 0.037 — permanently higher inflation
model.debasement(debt, model.shift_inflation(0.01)) / GDP_2012  # 0.019 — a flat +1pp
model.repression(debt, counterfactual_model(1), years=10)     # section 8
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

## Structure

```
src/debtmodel/
  curves.py        TermStructure — annual rate grids, interpolation, shifts
  maturity.py      DebtProfile — B_0^t, truncation, deferral (repression)
  burden.py        DebasementModel — ω_t, valuation, counterfactuals, repression
  calibration.py   US 2012 inputs and the paper's published targets
```

## The result

Inflation does not pay off the US public debt. At end-2012 composition, a
permanent +1pp shift in inflation at every horizon erodes about 1.9% of GDP, and
the paper's most aggressive credible counterfactual reaches 4.4%. The reason is
structural and reads directly off the model: erosion is roughly
`duration × value`, and with a 3.9-year duration on debt worth 51% of GDP that
is about 2% of GDP per percentage point. Getting to a large number requires
either inflation far outside what markets price as possible, or a long maturity
structure the Treasury does not have — which is why financial repression, an ex
post extension of maturity, is the one lever in the paper that moves the answer
by an order of magnitude.
