"""Derive the September 2026 payment profile B_0^t from primary sources.

Pulls Treasury MSPD table 3 (every marketable security outstanding) and the
New York Fed's SOMA holdings (what the central bank owns of each CUSIP), nets
one against the other, expands each security into its nominal payment stream,
and buckets those payments by year.

Run with network access to regenerate the literal array baked into
``debtmodel.calibration_2026``:

    uv run --with requests python scripts/build_2026_profile.py
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import urllib.request

MSPD = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1"
        "/debt/mspd/mspd_table_3?filter=record_date:eq:{date}&page[size]=10000")
SOMA = "https://markets.newyorkfed.org/api/soma/tsy/get/all/asof/{date}.json"

AS_OF = dt.date(2026, 9, 8)      # valuation date
MSPD_DATE = "2026-08-31"         # latest Monthly Statement of the Public Debt
SOMA_DATE = "2026-09-02"         # latest SOMA holdings snapshot

KIND = {
    "Bills Maturity Value": "bill",
    "Notes": "note",
    "Bonds": "bond",
    "Inflation-Protected Securities": "tips",
    "Floating Rate Notes": "frn",
}
HORIZON_YEARS = 30


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_outstanding() -> dict[str, dict]:
    """Every marketable security outstanding, aggregated over reopenings."""
    rows = _get(MSPD.format(date=MSPD_DATE))["data"]
    agg: dict[str, dict] = {}
    for r in rows:
        kind = KIND.get(r["security_class1_desc"])
        if kind is None:
            continue
        cusip = r["security_class2_desc"]
        a = agg.setdefault(
            cusip,
            dict(kind=kind, coupon=None, maturity=None, out=None, iss=0.0, red=0.0),
        )
        if r.get("maturity_date") not in (None, "null"):
            a["maturity"] = r["maturity_date"]
        if _num(r.get("interest_rate_pct")) is not None:
            a["coupon"] = _num(r["interest_rate_pct"])
        if _num(r.get("outstanding_amt")) is not None:
            a["out"] = _num(r["outstanding_amt"])
        a["iss"] += _num(r.get("issued_amt")) or 0.0
        a["red"] += _num(r.get("redeemed_amt")) or 0.0

    secs = {}
    for cusip, a in agg.items():
        # Reopened issues report the total only on the primary row; where it is
        # absent, issued less redeemed is the same quantity.
        amt = a["out"] if a["out"] is not None else a["iss"] - a["red"]
        if not a["maturity"] or amt <= 0:
            continue
        a["amount"] = amt  # $ millions
        secs[cusip] = a
    return secs


def load_fed_holdings() -> dict[str, float]:
    """Par value of each CUSIP held in the System Open Market Account, $mn."""
    rows = _get(SOMA.format(date=SOMA_DATE))["soma"]["holdings"]
    held: dict[str, float] = collections.defaultdict(float)
    for r in rows:
        par = _num(r.get("parValue"))
        if par:
            held[r["cusip"]] += par / 1e6
    return held


def payment_stream(sec: dict, as_of: dt.date):
    """Expand one security into ``(years_from_now, nominal $mn)`` payments.

    Notes and bonds pay a fixed semiannual coupon and repay principal at
    maturity; bills are a single payment of their maturity value.  Dates are
    stepped back from maturity in six-month increments, which reproduces the
    actual coupon calendar without needing the issue date.
    """
    mat = dt.date.fromisoformat(sec["maturity"])
    if mat <= as_of:
        return
    face = sec["amount"]

    if sec["kind"] in ("bill", "frn"):
        # A bill's outstanding amount is already its maturity value.  An FRN's
        # coupon resets to the 13-week bill rate, so beyond one reset its
        # payments are not fixed in nominal terms; it is handled by the caller.
        yield (mat - as_of).days / 365.25, face
        return

    coupon = (sec["coupon"] or 0.0) / 200.0 * face  # semiannual, percent of par
    dates, d = [], mat
    while d > as_of:
        dates.append(d)
        year, month = d.year, d.month - 6
        if month <= 0:
            year, month = year - 1, month + 12
        day = min(d.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30,
                          31, 31, 30, 31, 30, 31][month - 1])
        d = dt.date(year, month, day)
    for d in dates:
        amount = coupon + (face if d == mat else 0.0)
        yield (d - as_of).days / 365.25, amount


def main() -> None:
    secs = load_outstanding()
    fed = load_fed_holdings()

    totals = collections.Counter()
    for s in secs.values():
        totals[s["kind"]] += s["amount"]
    print("MSPD outstanding, $tn:")
    for k in ("bill", "note", "bond", "tips", "frn"):
        print(f"  {k:5s} {totals[k] / 1e6:7.3f}")
    print(f"Fed holds {sum(fed.values()) / 1e6:.3f}tn across {len(fed)} CUSIPs")

    # Annual buckets, at their midpoints (see calibration notes).
    buckets = [0.0] * HORIZON_YEARS
    excluded = collections.Counter()
    for cusip, s in secs.items():
        if s["kind"] == "tips":
            excluded["tips (real, not eroded by inflation)"] += s["amount"]
            continue
        private = s["amount"] - fed.get(cusip, 0.0)
        if private <= 0:
            continue
        scale = private / s["amount"]
        for t, amt in payment_stream(s, AS_OF):
            if s["kind"] == "frn":
                # Charge the FRN to its next quarterly reset, not its maturity:
                # its coupon floats, so its nominal payments are only fixed
                # over one reset period.
                t = 0.25
            i = int(t)
            if 0 <= i < HORIZON_YEARS:
                buckets[i] += amt * scale
    print(f"excluded: {dict(excluded)}")

    bn = [round(v / 1000.0, 1) for v in buckets]  # $bn
    print(f"\ntotal payments ${sum(bn) / 1000:.2f}tn over {HORIZON_YEARS}y")
    print("\n_PAYMENTS_2026 = np.array([")
    for i in range(0, HORIZON_YEARS, 6):
        print("    " + ", ".join(f"{v:9.1f}" for v in bn[i:i + 6]) + ",")
    print("], dtype=float)")


if __name__ == "__main__":
    main()
