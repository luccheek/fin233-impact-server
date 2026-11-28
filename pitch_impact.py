#!/usr/bin/env python3
"""
FIN 233 pitch-impact preview.

Before a vote: what would adding (or trimming) a candidate stock have done to
the portfolio's risk and return over the trailing window? Saves data/impact.json
and rebakes the dashboard's "Pitch impact" tab; the weight slider works
client-side, so the page stays a sendable, self-contained file.

Also writes data/impact_base.json (the portfolio/benchmark return series), which
impact_server.py uses to serve live "search any NYSE/Nasdaq ticker" lookups.

Usage:
    python3 pitch_impact.py --tickers AMD,SBUX          # today's pitches
    python3 pitch_impact.py --tickers NKE --years 5
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

from fetch_data import (HERE, SYMBOL_FIXES, bake_combined, build_detail,
                        clean_nan, get_info, get_news, load_cache,
                        load_news_cache, parse_rjf, save_cache, save_json)

BASE_PATH = os.path.join(HERE, "data", "impact_base.json")


def make_base(holdings, years, benchmark, rf, cache, px=None, save=True):
    """Portfolio + benchmark daily-return series (date-keyed) plus weights and
    sector mix — everything a candidate lookup needs besides the candidate
    itself. Baked by the CLI; reloaded (or rebuilt) by impact_server.py."""
    syms = [h["symbol"] for h in holdings]
    if px is None:
        print(f"Downloading {years}y prices for {len(syms)} holdings + {benchmark} ...")
        px = yf.download(sorted(set(syms + [benchmark])), period=f"{years}y",
                         interval="1d", auto_adjust=True, progress=False)["Close"].dropna(how="all")

    live = [h for h in holdings if h["symbol"] in px.columns
            and not px[h["symbol"]].dropna().empty]
    qty = {h["symbol"]: h["qty"] for h in live}
    port_val = (px[list(qty.keys())].ffill().bfill() * pd.Series(qty)).sum(axis=1)
    port_ret = port_val.pct_change().dropna()
    bench_ret = px[benchmark].ffill().pct_change().dropna()

    total_val = sum(h["value"] or 0 for h in live)
    weights = {h["symbol"]: (h["value"] or 0) / total_val for h in live}
    sectors = {}
    for h in live:
        sec = get_info(h["symbol"], cache).get("sector") or "Other"
        sectors[sec] = sectors.get(sec, 0) + weights[h["symbol"]]

    base = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "years": years, "benchmark": benchmark, "rf": rf,
        "totalValue": total_val,
        "sectors": sorted(({"name": k, "weight": v} for k, v in sectors.items()),
                          key=lambda s: -s["weight"]),
        "weights": {k: round(v, 6) for k, v in weights.items()},
        "portRet": {d.strftime("%Y-%m-%d"): round(float(v), 6)
                    for d, v in port_ret.items()},
        "benchRet": {d.strftime("%Y-%m-%d"): round(float(v), 6)
                     for d, v in bench_ret.items()},
    }
    if save:
        os.makedirs(os.path.dirname(BASE_PATH), exist_ok=True)
        with open(BASE_PATH, "w") as f:
            json.dump(clean_nan(base), f)
    return base


def build_candidate(sym, cpx, base, cache, news_cache, pm, light=False):
    """One candidate's full payload (impact series + detail drawer data).
    cpx: pandas Series of daily closes. Returns None if too little overlap.
    light=True skips the per-ticker news + earnings-calendar calls (used when
    pre-baking the ~600-name public search universe); the earnings date and
    consensus EPS then come from the Polymarket slug when a market exists."""
    cret = cpx.pct_change().dropna()
    cd = {d.strftime("%Y-%m-%d"): float(v) for d, v in cret.items()}
    common = sorted(set(cd) & set(base["portRet"]) & set(base["benchRet"]))
    if len(common) < 120:
        print(f"  ! {sym}: only {len(common)} overlapping days — skipped", file=sys.stderr)
        return None

    info = get_info(sym, cache)
    wk = cpx.iloc[-260:][::5]
    edate = eps = None
    if not light:
        try:
            cal = yf.Ticker(sym).calendar
            if isinstance(cal, dict):
                dts = cal.get("Earnings Date")
                if dts:
                    edate = str(dts[0])[:10]
                eps = cal.get("Earnings Average")
        except Exception:
            pass
    earn = None
    pmv = pm.get(sym)
    if pmv and edate is None:
        edate, eps = pmv.get("pmDate"), pmv.get("pmEps")
    if edate or pmv:
        earn = {"date": edate, "epsEst": eps}
        if pmv:
            earn.update({"pmProb": pmv.get("pmProb"), "pmUrl": pmv.get("pmUrl"),
                         "pmVol": pmv.get("pmVol")})
    held = base["weights"].get(sym, 0)
    return {
        "sym": sym,
        "name": info.get("shortName") or sym,
        "sector": info.get("sector") or "—",
        "held": round(held, 5),
        "price": info.get("currentPrice") or round(float(cpx.iloc[-1]), 2),
        "windowDays": len(common),
        "start": common[0], "end": common[-1],
        "dates": common,
        "portRet": [round(base["portRet"][d], 5) for d in common],
        "candRet": [round(cd[d], 5) for d in common],
        "benchRet": [round(base["benchRet"][d], 5) for d in common],
        "weekPx": [round(float(v), 2) for v in wk],
        "weekDates": [d.strftime("%Y-%m-%d") for d in wk.index],
        "detail": build_detail(info, [] if light else get_news(sym, news_cache),
                               round(float(cpx.max()), 2), round(float(cpx.min()), 2)),
        "earn": earn,
    }


def build(holdings, tickers, years, benchmark, rf, cache):
    syms = [h["symbol"] for h in holdings]
    cands = [SYMBOL_FIXES.get(t, t) for t in tickers]
    fetch = sorted(set(syms + cands + [benchmark]))
    print(f"Downloading {years}y prices for {len(fetch)} tickers ...")
    px = yf.download(fetch, period=f"{years}y", interval="1d",
                     auto_adjust=True, progress=False)["Close"].dropna(how="all")

    base = make_base(holdings, years, benchmark, rf, cache, px=px)

    try:
        from fetch_macro import build_polymarket
        pm = build_polymarket(set(cands))
    except Exception as e:
        print(f"  ! Polymarket lookup skipped: {e}", file=sys.stderr)
        pm = {}
    news_cache = load_news_cache()

    out = []
    for c in cands:
        if c not in px.columns or px[c].dropna().empty:
            print(f"  ! no price data for {c} — skipped", file=sys.stderr)
            continue
        cand = build_candidate(c, px[c].dropna(), base, cache, news_cache, pm)
        if cand:
            out.append(cand)
            print(f"  {c}: {cand['windowDays']} days, "
                  f"{'held at %.2f%%' % (100*cand['held']) if cand['held'] else 'not currently held'}")
    json.dump(news_cache, open(os.path.join(HERE, ".news_cache.json"), "w"))

    return {
        "asOf": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "years": years, "benchmark": benchmark, "rf": rf,
        "totalValue": base["totalValue"],
        "sectors": base["sectors"],
        "candidates": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", "--ticker", required=True,
                    help="comma-separated candidate tickers (today's pitches)")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--holdings", default=os.path.join(HERE, "holdings.csv"))
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--rf", type=float, default=4.0)
    args = ap.parse_args()

    cache = load_cache()
    holdings = parse_rjf(args.holdings)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    data = build(holdings, tickers, args.years, args.benchmark, args.rf, cache)
    save_cache(cache)
    if not data["candidates"]:
        sys.exit("No usable candidates — nothing baked.")
    save_json("impact.json", clean_nan(data))
    bake_combined()
    print("Pitch impact now lives in the dashboard's 'Pitch impact' tab.")


if __name__ == "__main__":
    main()
