#!/usr/bin/env python3
"""
FIN 233 macro briefing builder — the "Bloomberg lite" page.

Pulls macro series from FRED, a market snapshot from Yahoo Finance, and the
portfolio's upcoming earnings dates, then bakes dist/macro_briefing.html.

FRED key: env var FRED_API_KEY, or a .fred_key file next to this script
(free key from https://fred.stlouisfed.org/docs/api/api_key.html).

Usage:  python3 fetch_macro.py [--holdings holdings.csv]
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

import yfinance as yf

from fetch_data import HERE, SYMBOL_FIXES, bake_combined, clean_nan, parse_rjf, save_json

FRED_KEY_FILE = os.path.join(HERE, ".fred_key")
EARN_CACHE = os.path.join(HERE, ".earnings_cache.json")
EARN_TTL_DAYS = 3

# (fred_id, label, units-transform, format, group, teaching note)
SERIES = [
    ("DFEDTARU", "Fed funds target (upper)", "lin", "pct", "Rates & policy",
     "The anchor for every discount rate in the room"),
    ("DGS10", "10-year Treasury", "lin", "pct", "Rates & policy",
     "The usual risk-free rate in CAPM / WACC"),
    ("DGS2", "2-year Treasury", "lin", "pct", "Rates & policy",
     "Tracks where the market thinks the Fed goes next"),
    ("T10Y2Y", "10y − 2y spread", "lin", "pct", "Rates & policy",
     "Inverted curve = the classic recession signal"),
    ("DFII10", "10y real yield (TIPS)", "lin", "pct", "Rates & policy",
     "Real rates are what actually compete with stocks"),
    ("CPIAUCSL", "CPI inflation (YoY)", "pc1", "pct", "Inflation",
     "The headline print markets trade on"),
    ("PCEPILFE", "Core PCE (YoY)", "pc1", "pct", "Inflation",
     "What the Fed actually targets (2%)"),
    ("T5YIFR", "5y5y inflation expectations", "lin", "pct", "Inflation",
     "Market-implied long-run inflation"),
    ("DCOILWTICO", "WTI crude oil ($/bbl)", "lin", "num", "Inflation",
     "Energy prices feed straight into headline inflation"),
    ("PAYEMS", "Nonfarm payrolls (m/m)", "chg", "k", "Growth & labor",
     "The first-Friday jobs number"),
    ("UNRATE", "Unemployment rate", "lin", "pct", "Growth & labor",
     "Labor-market slack in one number"),
    ("A191RL1Q225SBEA", "Real GDP (q/q annualized)", "lin", "pct", "Growth & labor",
     "Quarterly reality check on \"the economy\""),
    ("RSAFS", "Retail sales (YoY)", "pc1", "pct", "Growth & labor",
     "The US consumer is ~2/3 of GDP"),
    ("VIXCLS", "VIX", "lin", "num", "Risk",
     "Options-implied volatility — the fear gauge (ch. 9)"),
    ("BAMLH0A0HYM2", "High-yield credit spread", "lin", "pct", "Risk",
     "Credit stress usually shows up before equity stress"),
]

CURVE = [("DGS1MO", "1M"), ("DGS3MO", "3M"), ("DGS6MO", "6M"), ("DGS1", "1Y"),
         ("DGS2", "2Y"), ("DGS3", "3Y"), ("DGS5", "5Y"), ("DGS7", "7Y"),
         ("DGS10", "10Y"), ("DGS20", "20Y"), ("DGS30", "30Y")]

INDEXES = [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("^DJI", "Dow Jones"),
           ("IWM", "Russell 2000"), ("GC=F", "Gold"),
           ("DX-Y.NYB", "Dollar index"), ("BTC-USD", "Bitcoin")]

SECTORS = [("XLK", "Tech"), ("XLF", "Financials"), ("XLV", "Health"),
           ("XLE", "Energy"), ("XLI", "Industrials"), ("XLY", "Discretionary"),
           ("XLP", "Staples"), ("XLU", "Utilities"), ("XLB", "Materials"),
           ("XLRE", "Real estate"), ("XLC", "Comms")]


def fred_key():
    k = os.environ.get("FRED_API_KEY")
    if k:
        return k.strip()
    if os.path.exists(FRED_KEY_FILE):
        return open(FRED_KEY_FILE).read().strip()
    sys.exit("No FRED API key. Set FRED_API_KEY or put the key in .fred_key")


def fred_obs(sid, key, start, units="lin"):
    url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode({
        "series_id": sid, "api_key": key, "file_type": "json",
        "observation_start": start, "units": units})
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    return [[o["date"], float(o["value"])]
            for o in data.get("observations", [])
            if o["value"] not in (".", "")]


def thin(points, keep=120):
    if len(points) <= keep:
        return points
    step = len(points) / keep
    out = [points[int(i * step)] for i in range(keep)]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def build_series(key):
    start = (date.today() - timedelta(days=730)).isoformat()
    out = []
    for sid, label, units, fmt, group, note in SERIES:
        try:
            pts = fred_obs(sid, key, start, units)
        except Exception as e:
            print(f"  ! FRED {sid} failed: {e}", file=sys.stderr)
            continue
        if len(pts) < 2:
            print(f"  ! FRED {sid}: not enough data", file=sys.stderr)
            continue
        out.append({
            "id": sid, "label": label, "fmt": fmt, "group": group, "note": note,
            "latest": pts[-1][1], "latestDate": pts[-1][0], "prior": pts[-2][1],
            "points": thin(pts),
        })
        print(f"  {sid}: {pts[-1][1]} ({pts[-1][0]})")
    return out


def build_curve(key):
    today, year_ago = [], []
    start = (date.today() - timedelta(days=400)).isoformat()
    for sid, label in CURVE:
        try:
            pts = fred_obs(sid, key, start)
        except Exception as e:
            print(f"  ! FRED {sid} failed: {e}", file=sys.stderr)
            continue
        if not pts:
            continue
        today.append([label, pts[-1][1]])
        # first observation on/after ~1 year ago
        cutoff = (date.today() - timedelta(days=365)).isoformat()
        past = [p for p in pts if p[0] >= cutoff]
        if past:
            year_ago.append([label, past[0][1]])
    return {"today": today, "yearAgo": year_ago,
            "date": datetime.now().strftime("%Y-%m-%d")}


def build_markets():
    tickers = [t for t, _ in INDEXES + SECTORS]
    print(f"Downloading market snapshot for {len(tickers)} tickers ...")
    # no global ffill: BTC-USD trades weekends, and filling equities into those
    # rows would make "last two closes" identical (1d change = 0)
    px = yf.download(tickers, period="1y", interval="1d",
                     auto_adjust=True, progress=False)["Close"]

    def stats(t):
        s = px[t].dropna()
        if len(s) < 3:
            return None, None, None
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        this_year = s[s.index.year == date.today().year]
        prior_year = s[s.index.year < date.today().year]
        base = float(prior_year.iloc[-1]) if len(prior_year) else float(this_year.iloc[0])
        return last, last / prev - 1, last / base - 1

    def rows(pairs):
        out = []
        for t, name in pairs:
            last, d1, ytd = stats(t)
            if last is None:
                print(f"  ! no data for {t}", file=sys.stderr)
                continue
            out.append({"sym": t, "name": name, "last": round(last, 2),
                        "d1": d1, "ytd": ytd})
        return out

    return rows(INDEXES), rows(SECTORS)


def first_fridays(months_ahead=5):
    """Jobs report lands on the first Friday of the month (usually)."""
    out, d = [], date.today().replace(day=1)
    for _ in range(months_ahead + 1):
        ff = d + timedelta(days=(4 - d.weekday()) % 7)
        if ff >= date.today():
            out.append({"date": ff.isoformat(),
                        "event": "Jobs report (BLS employment situation)",
                        "approx": True})
        d = (d + timedelta(days=32)).replace(day=1)
    return out


def load_calendar(path):
    rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if (r.get("date") or "") >= date.today().isoformat():
                    rows.append({"date": r["date"], "event": r["event"],
                                 "approx": (r.get("approx") or "").strip().upper() == "Y"})
    rows += first_fridays()
    rows.sort(key=lambda r: r["date"])
    return rows[:14]


def build_earnings(holdings):
    cache = {}
    if os.path.exists(EARN_CACHE):
        try:
            cache = json.load(open(EARN_CACHE))
        except Exception:
            pass
    print(f"Fetching earnings dates + analyst estimates for {len(holdings)} holdings (cached {EARN_TTL_DAYS}d) ...")
    out = []
    for i, h in enumerate(holdings):
        t = h["symbol"]
        ent = cache.get(t)
        # "eps" key marks the current cache schema; older entries refetch
        if not ent or "eps" not in ent or time.time() - ent.get("_ts", 0) > EARN_TTL_DAYS * 86400:
            d = eps = rev = None
            try:
                cal = yf.Ticker(t).calendar
                if isinstance(cal, dict):
                    dates = cal.get("Earnings Date")
                    if dates:
                        d = str(dates[0])[:10]
                    eps = cal.get("Earnings Average")
                    rev = cal.get("Revenue Average")
            except Exception:
                pass
            ent = {"date": d, "eps": eps, "rev": rev, "_ts": time.time()}
            cache[t] = ent
        if ent["date"]:
            out.append({"sym": t, "name": h["name"], "date": ent["date"],
                        "epsEst": ent.get("eps"), "revEst": ent.get("rev")})
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(holdings)}")
            json.dump(cache, open(EARN_CACHE, "w"))
    json.dump(cache, open(EARN_CACHE, "w"))
    horizon = (date.today() + timedelta(days=21)).isoformat()
    upcoming = sorted(
        [e for e in out if date.today().isoformat() <= e["date"] <= horizon],
        key=lambda e: e["date"])
    print(f"  {len(upcoming)} holdings report in the next 3 weeks")
    return upcoming


def build_polymarket(symbols):
    """Polymarket earnings-beat markets, keyed by ticker. Slugs look like
    jpm-quarterly-earnings-gaap-eps-07-14-2026-5pt68 — ticker up front.
    symbols=None returns every earnings market (used by impact_server)."""
    url = ("https://gamma-api.polymarket.com/events?"
           + urllib.parse.urlencode({"tag_slug": "earnings", "closed": "false",
                                     "limit": 500}))
    try:
        # Polymarket's CDN rejects urllib's default user agent
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.4.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            events = json.load(r)
    except Exception as e:
        print(f"  ! Polymarket fetch failed: {e}", file=sys.stderr)
        return {}
    out = {}
    for e in events:
        slug = e.get("slug") or ""
        if "-quarterly-earnings" not in slug:
            continue
        tick = slug.split("-quarterly-earnings")[0].upper()
        if symbols is not None and tick not in symbols:
            continue
        m = (e.get("markets") or [{}])[0]
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
            outcomes = json.loads(m.get("outcomes") or "[]")
            yes_i = outcomes.index("Yes") if "Yes" in outcomes else 0
            prob = float(prices[yes_i])
        except Exception:
            continue
        # slug embeds the earnings date and consensus strike:
        # hog-quarterly-earnings-gaap-eps-07-23-2026-0pt62
        pm_date = pm_eps = None
        md = re.search(r"-(\d{2})-(\d{2})-(\d{4})-", slug)
        if md:
            pm_date = f"{md.group(3)}-{md.group(1)}-{md.group(2)}"
        ms = re.search(r"-(neg)?(\d+)pt(\d+)$", slug)
        if ms:
            pm_eps = float(f"{ms.group(2)}.{ms.group(3)}") * (-1 if ms.group(1) else 1)
        out[tick] = {"pmProb": prob, "pmVol": e.get("volume"),
                     "pmUrl": "https://polymarket.com/event/" + slug,
                     "pmTitle": e.get("title"),
                     "pmDate": pm_date, "pmEps": pm_eps}
    print(f"  Polymarket: matched {len(out)} earnings markets to holdings")
    return out


def build_polymarket_macro():
    """Fed-decision / rate-path / recession markets for the macro tab."""
    def get(url):
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.4.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def outcomes(e, max_n=4):
        outs = []
        for m in e.get("markets") or []:
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
                names = json.loads(m.get("outcomes") or "[]")
                yes = float(prices[names.index("Yes")]) if "Yes" in names else float(prices[0])
            except Exception:
                continue
            outs.append({"label": m.get("groupItemTitle") or "Yes", "prob": yes})
        outs.sort(key=lambda x: -x["prob"])
        return outs[:max_n]

    def card(e):
        return {"title": e.get("title"), "endDate": (e.get("endDate") or "")[:10],
                "url": "https://polymarket.com/event/" + (e.get("slug") or ""),
                "volume": e.get("volume"), "outcomes": outcomes(e)}

    cards = []
    try:
        evs = get("https://gamma-api.polymarket.com/events?tag_slug=fed-rates"
                  "&closed=false&limit=100")
    except Exception as e:
        print(f"  ! Polymarket fed-rates fetch failed: {e}", file=sys.stderr)
        evs = []

    def earliest(prefix):
        matches = [e for e in evs if (e.get("slug") or "").startswith(prefix)]
        matches.sort(key=lambda e: e.get("endDate") or "9999")
        return matches[0] if matches else None

    for prefix in ("fed-decision-in", "how-many-fed-rate-cuts-in",
                   "fed-rate-hike-in"):
        e = earliest(prefix)
        if e:
            cards.append(card(e))
    try:
        d = get("https://gamma-api.polymarket.com/public-search?q=us+recession"
                "&limit_per_type=10&events_status=active")
        e = next((x for x in d.get("events", [])
                  if (x.get("slug") or "").startswith("us-recession")), None)
        if e:
            cards.append(card(e))
    except Exception as e:
        print(f"  ! Polymarket recession fetch failed: {e}", file=sys.stderr)
    print(f"  Polymarket macro: {len(cards)} markets")
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", default=os.path.join(HERE, "holdings.csv"))
    ap.add_argument("--calendar", default=os.path.join(HERE, "calendar.csv"))
    ap.add_argument("--skip-earnings", action="store_true",
                    help="skip the slow per-holding earnings-date fetch")
    args = ap.parse_args()

    key = fred_key()
    print("Fetching FRED series ...")
    series = build_series(key)
    print("Fetching yield curve ...")
    curve = build_curve(key)
    indexes, sectors = build_markets()
    calendar = load_calendar(args.calendar)
    earnings = []
    if not args.skip_earnings and os.path.exists(args.holdings):
        holdings = parse_rjf(args.holdings)
        earnings = build_earnings(holdings)
        print("Fetching Polymarket earnings markets ...")
        pm = build_polymarket({h["symbol"] for h in holdings})
        for e in earnings:
            e.update(pm.get(e["sym"], {}))
    else:
        # --skip-earnings: keep the earnings panel from the last full run
        prev = os.path.join(HERE, "data", "macro.json")
        if os.path.exists(prev):
            with open(prev) as f:
                earnings = json.load(f).get("earnings") or []
            print(f"  keeping {len(earnings)} earnings rows from the last full run")

    print("Fetching Polymarket macro markets ...")
    pm_macro = build_polymarket_macro()

    save_json("macro.json", clean_nan({
        "asOf": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "series": series, "curve": curve,
        "indexes": indexes, "sectors": sectors,
        "calendar": calendar, "earnings": earnings,
        "pmMacro": pm_macro,
    }))
    bake_combined()
    print("Done. Macro now lives in the portfolio dashboard's 'Market & macro' tab.")


if __name__ == "__main__":
    main()
