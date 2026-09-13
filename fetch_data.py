#!/usr/bin/env python3
"""
FIN 233 tools — data fetcher & page builder.

Reads a holdings CSV (Raymond James export today; Capital IQ later), pulls
prices/fundamentals from Yahoo Finance, and bakes the data into the HTML
templates so the pages in dist/ are fully self-contained (no server, no CDN).

Usage:
    python3 fetch_data.py                          # defaults below
    python3 fetch_data.py --holdings holdings.csv \
        --comps "AAPL:MSFT,GOOGL,META,NVDA" \
        --dcf AAPL,MSFT,NVDA,KO,NKE \
        --benchmark SPY --rf 4.0
"""

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import urllib.request
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "templates")
DIST = os.path.join(HERE, "dist")
CACHE_PATH = os.path.join(HERE, ".info_cache.json")
CACHE_TTL_DAYS = 7

# Pin yfinance's sqlite timezone cache to a project-local path. Under a
# scheduled (launchd) run the default user-cache location produced
# "unable to open database file", which made bulk downloads return empty
# columns. Every script imports this module, so this applies pipeline-wide.
try:
    _YF_CACHE = os.path.join(HERE, ".yf_cache")
    os.makedirs(_YF_CACHE, exist_ok=True)
    yf.set_tz_cache_location(_YF_CACHE)
except Exception as _e:  # non-fatal: fall back to yfinance's default
    print(f"  ! could not set yfinance cache location: {_e}", file=sys.stderr)

# Symbols the broker export writes differently than Yahoo expects,
# plus CUSIPs that appear in place of tickers.
SYMBOL_FIXES = {
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
    "30231G102": "XOM",  # Exxon Mobil CUSIP
}

EQUITY_PRODUCT_TYPES = {"Stock", "Real Estate Investment Trusts", "Partnerships"}

# Tickers whose Yahoo profile is missing a sector.
SECTOR_OVERRIDES = {
    "RTX": "Industrials",
    "FISV": "Technology",
}


def parse_money(s):
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").replace("*", "").replace("^", "").strip()
    if s in ("", "-", "N/A"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def parse_rjf(path):
    """Raymond James portfolio export -> list of holdings dicts."""
    holdings = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ptype = (row.get("Product Type") or "").strip()
            if ptype not in EQUITY_PRODUCT_TYPES:
                continue
            raw_sym = (row.get("SYMBOL/CUSIP") or "").strip()
            if not raw_sym:
                continue
            sym = SYMBOL_FIXES.get(raw_sym, raw_sym)
            qty = parse_money(row.get("Quantity"))
            if not qty:
                continue
            holdings.append({
                "symbol": sym,
                "name": (row.get("Description") or sym).strip().title(),
                "qty": qty,
                "value": parse_money(row.get("Current Value")),
                "invested": parse_money(row.get("Amount Invested (†)")),
            })
    return holdings


def parse_ciq(path):
    """Capital IQ export parser — to be implemented once the export format is in hand."""
    raise NotImplementedError(
        "Capital IQ parsing not implemented yet. Export holdings as the RJF-style "
        "CSV for now, or extend parse_ciq() with the CIQ column names."
    )


# ---------------------------------------------------------------- info cache

def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


INFO_FIELDS = [
    "sector", "industry", "shortName", "currentPrice", "marketCap",
    "trailingPE", "forwardPE", "enterpriseToEbitda",
    "priceToSalesTrailing12Months", "priceToBook",
    "profitMargins", "operatingMargins", "grossMargins", "revenueGrowth",
    "totalRevenue", "ebitda", "netIncomeToCommon", "trailingEps",
    "sharesOutstanding", "totalDebt", "totalCash", "bookValue", "beta",
    # analyst + descriptive fields for the stock-detail drawer (same .info call,
    # no extra requests)
    "targetMeanPrice", "targetHighPrice", "targetLowPrice",
    "recommendationMean", "recommendationKey", "numberOfAnalystOpinions",
    "forwardEps", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "longBusinessSummary",
]

NEWS_CACHE = os.path.join(HERE, ".news_cache.json")
NEWS_TTL_DAYS = 1


def get_news(ticker, cache, limit=3):
    ent = cache.get(ticker)
    if ent and time.time() - ent.get("_ts", 0) < NEWS_TTL_DAYS * 86400:
        return ent["items"]
    items = []
    try:
        for it in (yf.Ticker(ticker).news or [])[:12]:
            c = it.get("content") or it
            title = c.get("title")
            url = ((c.get("canonicalUrl") or {}).get("url")
                   or (c.get("clickThroughUrl") or {}).get("url") or it.get("link"))
            pub = (c.get("pubDate") or c.get("providerPublishTime") or "")
            provider = ((c.get("provider") or {}).get("displayName")
                        if isinstance(c.get("provider"), dict) else it.get("publisher")) or ""
            if title:
                items.append({"title": title, "url": url,
                              "date": str(pub)[:10], "src": provider})
            if len(items) >= limit:
                break
    except Exception as e:
        print(f"  ! news fetch failed for {ticker}: {e}", file=sys.stderr)
    cache[ticker] = {"items": items, "_ts": time.time()}
    return items


def get_info(ticker, cache):
    ent = cache.get(ticker)
    # "targetMeanPrice" marks the current schema; older cached entries refetch
    if (ent and "targetMeanPrice" in ent
            and time.time() - ent.get("_ts", 0) < CACHE_TTL_DAYS * 86400):
        return ent
    try:
        raw = yf.Ticker(ticker).info or {}
    except Exception as e:
        print(f"  ! info fetch failed for {ticker}: {e}", file=sys.stderr)
        raw = {}
    ent = {k: raw.get(k) for k in INFO_FIELDS}
    ent["_ts"] = time.time()
    if any(v is not None for k, v in ent.items() if k != "_ts"):
        cache[ticker] = ent  # don't cache total failures — retry next run
    return ent


# ------------------------------------------------------------------ metrics

def max_drawdown(series):
    peak, mdd = -math.inf, 0.0
    for v in series:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return mdd


def regress(y, x):
    """OLS slope/intercept of y on x (daily returns). Returns (beta, alpha_daily)."""
    n = len(x)
    if n < 20:
        return None, None
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    var = sum((a - mx) ** 2 for a in x) / n
    if var == 0:
        return None, None
    beta = cov / var
    alpha = my - beta * mx
    return beta, alpha


BENCHMARKS = [("SPY", "S&P 500"), ("RSP", "S&P 500 equal-weight"),
              ("VT", "Total world stock market")]


def top_headline(ticker):
    try:
        for item in yf.Ticker(ticker).news or []:
            c = item.get("content") or item
            title = c.get("title")
            url = ((c.get("canonicalUrl") or {}).get("url")
                   or (c.get("clickThroughUrl") or {}).get("url")
                   or item.get("link"))
            if title:
                return title, url
    except Exception:
        pass
    return None, None


def build_portfolio_data(holdings, benchmarks, rf_annual):
    tickers = [h["symbol"] for h in holdings]
    bsyms = [b[0] for b in benchmarks]
    print(f"Downloading 1y prices for {len(tickers)} holdings + {', '.join(bsyms)} ...")
    px = yf.download(tickers + bsyms, period="1y", interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    px = px.ffill().dropna(axis=0, how="all")

    missing = [t for t in tickers if t not in px.columns or px[t].dropna().empty]
    if missing:
        print(f"  ! no price history for: {', '.join(missing)} (excluded)", file=sys.stderr)
    live = [h for h in holdings if h["symbol"] not in missing]

    qty = {h["symbol"]: h["qty"] for h in live}
    sub = px[list(qty.keys())].ffill().bfill()
    port_val = (sub * pd.Series(qty)).sum(axis=1)
    port_ret = port_val.pct_change().dropna()

    rf_daily = rf_annual / 100 / 252
    ex = port_ret - rf_daily
    sharpe = (ex.mean() / ex.std() * math.sqrt(252)) if ex.std() > 0 else None
    port_cum = (port_val / port_val.iloc[0] - 1)

    # one entry per benchmark: cumulative series + relative stats
    bench_list = []
    for sym, name in benchmarks:
        if sym not in px.columns or px[sym].dropna().empty:
            print(f"  ! benchmark {sym} has no data — skipped", file=sys.stderr)
            continue
        b = px[sym].ffill().bfill()
        br = b.pct_change().dropna()
        idx = port_ret.index.intersection(br.index)
        beta, alpha_d = regress(list(port_ret[idx] - rf_daily), list(br[idx] - rf_daily))
        bcum = b / b.iloc[0] - 1
        bench_list.append({
            "sym": sym, "name": name,
            "cum": [round(float(v), 5) for v in bcum],
            "ret": float(bcum.iloc[-1]),
            "mdd": max_drawdown(list(b / b.iloc[0])),
            "alpha": (alpha_d * 252) if alpha_d is not None else None,
            "beta": beta,
        })

    # shared weekly date axis for per-stock detail sparklines
    weekly_idx = sub.index[::5]
    week_dates = [d.strftime("%Y-%m-%d") for d in weekly_idx]

    # per-position period stats + 5-day returns + weekly price series
    total_val = sum(h["value"] or 0 for h in live)
    positions = []
    for h in live:
        s = sub[h["symbol"]]
        period_ret = s.iloc[-1] / s.iloc[0] - 1
        w = (h["value"] or 0) / total_val if total_val else 0
        positions.append({
            "symbol": h["symbol"], "name": h["name"], "qty": h["qty"],
            "price": round(float(s.iloc[-1]), 2),
            "value": h["value"], "invested": h["invested"],
            "weight": w, "periodReturn": float(period_ret),
            "contribution": w * float(period_ret),
            "ret5": float(s.iloc[-1] / s.iloc[-6] - 1) if len(s) > 6 else None,
            "weekPx": [round(float(v), 2) for v in s.reindex(weekly_idx).ffill()],
            "high52": round(float(s.max()), 2), "low52": round(float(s.min()), 2),
        })
    positions.sort(key=lambda p: -(p["value"] or 0))

    # week-over-week movers with a headline each
    ranked = sorted((p for p in positions if p["ret5"] is not None),
                    key=lambda p: -p["ret5"])
    up, down = ranked[:5], ranked[-5:][::-1]
    print(f"Fetching headlines for {len(up) + len(down)} movers ...")
    movers = {"up": [], "down": []}
    for side, rows in (("up", up), ("down", down)):
        for p in rows:
            title, url = top_headline(p["symbol"])
            movers[side].append({"sym": p["symbol"], "name": p["name"],
                                 "ret5": p["ret5"], "headline": title, "url": url})

    dates = [d.strftime("%Y-%m-%d") for d in port_cum.index]
    return live, {
        "asOf": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "benchmark": bench_list[0]["sym"] if bench_list else None,
        "benchmarks": bench_list,
        "rfAnnual": rf_annual,
        "totalValue": total_val,
        "totalInvested": sum(h["invested"] or 0 for h in live),
        "periodReturn": float(port_cum.iloc[-1]),
        "sharpe": sharpe,
        "maxDrawdown": max_drawdown(list(port_val / port_val.iloc[0])),
        "dates": dates,
        "weekDates": week_dates,
        "portCum": [round(float(v), 5) for v in port_cum],
        "positions": positions,
        "movers": movers,
        "excluded": missing,
        # internal: daily portfolio + SPY returns for factor/risk work (popped before save)
        "_daily": [[d.strftime("%Y-%m-%d"), float(v)] for d, v in port_ret.items()],
        "_spyDaily": ([[d.strftime("%Y-%m-%d"), float(v)]
                       for d, v in px["SPY"].ffill().bfill().pct_change().dropna().items()]
                      if "SPY" in px.columns else []),
    }


def build_detail(info, news, fallback_hi=None, fallback_lo=None):
    """Normalized per-stock detail object, shared by the holdings drawer and the
    pitch-impact candidate drawer."""
    return {
        "industry": info.get("industry"),
        "summary": (info.get("longBusinessSummary") or "")[:600] or None,
        "marketCap": info.get("marketCap"),
        "beta": info.get("beta"),
        "pe": info.get("trailingPE"), "fpe": info.get("forwardPE"),
        "evEbitda": info.get("enterpriseToEbitda"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "pb": info.get("priceToBook"),
        "grossMargin": info.get("grossMargins"),
        "opMargin": info.get("operatingMargins"),
        "netMargin": info.get("profitMargins"),
        "revGrowth": info.get("revenueGrowth"),
        "divYield": info.get("dividendYield"),
        "fwdEps": info.get("forwardEps"), "trailEps": info.get("trailingEps"),
        "high52": info.get("fiftyTwoWeekHigh") or fallback_hi,
        "low52": info.get("fiftyTwoWeekLow") or fallback_lo,
        "target": {"mean": info.get("targetMeanPrice"),
                   "high": info.get("targetHighPrice"),
                   "low": info.get("targetLowPrice")},
        "recKey": info.get("recommendationKey"),
        "recMean": info.get("recommendationMean"),
        "numAnalysts": info.get("numberOfAnalystOpinions"),
        "news": news,
    }


def load_news_cache():
    if os.path.exists(NEWS_CACHE):
        try:
            return json.load(open(NEWS_CACHE))
        except Exception:
            pass
    return {}


def enrich_details(data, cache, fetch_news=True):
    """Attach per-stock detail (analyst targets, fundamentals, news) used by the
    holdings drawer. Analyst fields are already in the cached .info; only news
    is a fresh call, cached daily."""
    news_cache = load_news_cache()
    n = len(data["positions"])
    if fetch_news:
        print(f"Fetching headlines + analyst detail for {n} holdings "
              f"(cached {NEWS_TTL_DAYS}d) ...")
    for i, p in enumerate(data["positions"]):
        info = cache.get(p["symbol"], {})
        news = (get_news(p["symbol"], news_cache) if fetch_news
                else news_cache.get(p["symbol"], {}).get("items", []))
        p["detail"] = build_detail(info, news, p.get("high52"), p.get("low52"))
        if fetch_news and (i + 1) % 30 == 0:
            print(f"  {i + 1}/{n}")
            json.dump(news_cache, open(NEWS_CACHE, "w"))
    json.dump(news_cache, open(NEWS_CACHE, "w"))


def add_sectors(data, holdings, cache):
    print(f"Fetching sector info for {len(holdings)} tickers (cached after first run) ...")
    sector_by_symbol = {}
    for i, h in enumerate(holdings):
        info = get_info(h["symbol"], cache)
        sector_by_symbol[h["symbol"]] = (
            info.get("sector") or SECTOR_OVERRIDES.get(h["symbol"]) or "Other")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(holdings)}")
            save_cache(cache)
    sectors = {}
    for p in data["positions"]:
        sec = sector_by_symbol.get(p["symbol"], "Other")
        p["sector"] = sec
        sectors[sec] = sectors.get(sec, 0) + (p["value"] or 0)
    data["sectors"] = sorted(
        [{"name": k, "value": v} for k, v in sectors.items()],
        key=lambda s: -s["value"])


# ------------------------------------------------------------------- comps

def build_comps_data(subject, peers, cache):
    tickers = [subject] + peers
    print(f"Fetching comps stats for {', '.join(tickers)} ...")
    companies = []
    for t in tickers:
        i = get_info(t, cache)
        shares = i.get("sharesOutstanding")
        companies.append({
            "symbol": t,
            "name": i.get("shortName") or t,
            "isSubject": t == subject,
            "price": i.get("currentPrice"),
            "marketCap": i.get("marketCap"),
            "trailingPE": i.get("trailingPE"),
            "forwardPE": i.get("forwardPE"),
            "evEbitda": i.get("enterpriseToEbitda"),
            "priceSales": i.get("priceToSalesTrailing12Months"),
            "priceBook": i.get("priceToBook"),
            "grossMargin": i.get("grossMargins"),
            "opMargin": i.get("operatingMargins"),
            "netMargin": i.get("profitMargins"),
            "revGrowth": i.get("revenueGrowth"),
            "eps": i.get("trailingEps"),
            "revenue": i.get("totalRevenue"),
            "ebitda": i.get("ebitda"),
            "bookValue": i.get("bookValue"),  # per share
            "shares": shares,
            "netDebt": (i.get("totalDebt") or 0) - (i.get("totalCash") or 0),
        })
    return {"asOf": datetime.now().strftime("%Y-%m-%d %H:%M"), "companies": companies}


def build_dcf_presets(tickers, cache):
    print(f"Fetching DCF presets for {', '.join(tickers)} ...")
    presets = []
    for t in tickers:
        i = get_info(t, cache)
        if not i.get("totalRevenue") or not i.get("sharesOutstanding"):
            print(f"  ! skipping preset {t}: missing revenue/shares", file=sys.stderr)
            continue
        presets.append({
            "symbol": t,
            "name": i.get("shortName") or t,
            "revenue": i["totalRevenue"],
            "ebitMargin": i.get("operatingMargins") or 0.15,
            "revGrowth": i.get("revenueGrowth") or 0.05,
            "shares": i["sharesOutstanding"],
            "netDebt": (i.get("totalDebt") or 0) - (i.get("totalCash") or 0),
            "price": i.get("currentPrice"),
            "beta": i.get("beta"),
        })
    return presets


# ---------------------------------------------------------- factors & risk

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

# S&P 500 sector weights (%), approximate — update once a semester.
# Names use Yahoo's sector labels so they line up with the holdings.
SP500_SECTOR_WEIGHTS = {
    "Technology": 33.5, "Financial Services": 13.5, "Consumer Cyclical": 10.5,
    "Communication Services": 10.0, "Healthcare": 9.5, "Industrials": 8.5,
    "Consumer Defensive": 5.5, "Energy": 3.0, "Utilities": 2.5,
    "Real Estate": 2.0, "Basic Materials": 1.5,
}


def load_ff5():
    """Ken French daily 5-factor data -> {date: [MktRF, SMB, HML, RMW, CMA, RF]}
    as decimal returns. Cached for 7 days (the library updates monthly)."""
    path = os.path.join(DATA_DIR, "ff5_daily.csv")
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > 7 * 86400:
        print("Downloading Fama-French 5-factor daily data ...")
        req = urllib.request.Request(FF5_URL, headers={"User-Agent": "curl/8.4.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            raw = z.read(z.namelist()[0]).decode("latin-1")
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write(raw)
    out = {}
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7 and len(parts[0]) == 8 and parts[0].isdigit():
                try:
                    vals = [float(x) / 100 for x in parts[1:7]]
                except ValueError:
                    continue
                out[f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}"] = vals
    return out


def ols(y, cols):
    """OLS with intercept. Returns (coefs, t-stats, r2); coefs[0] is alpha."""
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    tss = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1 - float(resid @ resid) / tss if tss else 0.0
    return beta, beta / se, r2


def build_factor_models(daily, spy_daily):
    """CAPM / FF3 / FF5 regressions, kept consistent with the scorecard:
    excess returns use the Fama-French daily T-bill, and the market leg is
    SPY's excess return (not FF's total-market factor) so the CAPM alpha here
    equals the headline 'Alpha vs SPY' card exactly. SMB/HML/RMW/CMA come from
    Ken French. daily/spy_daily are [[date, return], ...]."""
    try:
        ff = load_ff5()
    except Exception as e:
        print(f"  ! Fama-French download failed: {e}", file=sys.stderr)
        return None
    spy = dict(spy_daily)
    rows = []
    for d, r in daily:
        if d in ff and d in spy:
            rf = ff[d][5]
            mkt = spy[d] - rf  # SPY excess return as the market factor
            rows.append((r - rf, [mkt, ff[d][1], ff[d][2], ff[d][3], ff[d][4]], d))
    if len(rows) < 60:
        print(f"  ! only {len(rows)} days overlap Fama-French data — skipping factors",
              file=sys.stderr)
        return None
    y = np.array([r[0] for r in rows])
    F = np.array([r[1] for r in rows])
    names = ["Mkt (SPY)", "SMB", "HML", "RMW", "CMA"]
    models = []
    for label, k in (("CAPM", 1), ("FF 3-factor", 3), ("FF 5-factor", 5)):
        beta, t, r2 = ols(y, [F[:, i] for i in range(k)])
        models.append({
            "name": label,
            "alphaAnnual": float(beta[0] * 252), "alphaT": float(t[0]), "r2": float(r2),
            "betas": {names[i]: [float(beta[i + 1]), float(t[i + 1])] for i in range(k)},
        })
    # Sharpe from the same excess-return series (daily T-bill), same window —
    # keeps the scorecard consistent with the factor table.
    sharpe = float(y.mean() / y.std() * math.sqrt(252)) if y.std() > 0 else None
    print(f"  factor regressions on {len(rows)} days "
          f"({rows[0][2]} → {rows[-1][2]}); FF5 R² = {models[2]['r2']:.2f}; "
          f"Sharpe (T-bill) = {sharpe:.2f}")
    return {"start": rows[0][2], "end": rows[-1][2], "n": len(rows),
            "factorNames": names, "models": models, "sharpe": sharpe}


def add_risk(data, daily, cache):
    rets = np.array([r for _, r in daily])
    weights = sorted((p["weight"] for p in data["positions"]), reverse=True)

    pe_pairs = [(p["weight"], cache.get(p["symbol"], {}).get("trailingPE"))
                for p in data["positions"]]
    valid = [(w, pe) for w, pe in pe_pairs if pe and pe > 0]
    port_pe = None
    if valid:
        tot = sum(w for w, _ in valid)
        # harmonic mean = value-weighted earnings yield, the right way to aggregate P/E
        port_pe = tot / sum(w / pe for w, pe in valid)

    total_sec = sum(s["value"] for s in data["sectors"]) or 1
    active, seen = [], set()
    for s in data["sectors"]:
        pw = 100 * s["value"] / total_sec
        spx = SP500_SECTOR_WEIGHTS.get(s["name"])
        active.append({"name": s["name"], "port": pw, "spx": spx,
                       "diff": (pw - spx) if spx is not None else None})
        seen.add(s["name"])
    for name, spx in SP500_SECTOR_WEIGHTS.items():
        if name not in seen:
            active.append({"name": name, "port": 0.0, "spx": spx, "diff": -spx})
    active.sort(key=lambda a: -(a["diff"] if a["diff"] is not None else -999))

    data["risk"] = {
        "annVol": float(rets.std() * math.sqrt(252)),
        "var95": float(np.percentile(rets, 5)),
        "var95Dollar": float(-np.percentile(rets, 5) * data["totalValue"]),
        "topWeight": weights[0] if weights else None,
        "top10Weight": sum(weights[:10]),
        "effectiveN": 1 / sum(w * w for w in weights) if weights else None,
        "portPE": port_pe,
        "spyPE": get_info("SPY", cache).get("trailingPE"),
        "activeSectors": active,
    }


# ------------------------------------------------------------------ pitches

def parse_pitches(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if (r.get("ticker") or "").strip()]


def build_pitch_data(rows, benchmark):
    tickers = sorted({SYMBOL_FIXES.get(r["ticker"].strip().upper(), r["ticker"].strip().upper())
                      for r in rows})
    start = min(r["date"] for r in rows)
    print(f"Downloading pitch prices for {', '.join(tickers)} since {start} ...")
    px = yf.download(tickers + [benchmark], start=start, interval="1d",
                     auto_adjust=True, progress=False)["Close"].ffill()

    pitches = []
    for r in rows:
        t = SYMBOL_FIXES.get(r["ticker"].strip().upper(), r["ticker"].strip().upper())
        if t not in px.columns or px[t].dropna().empty:
            print(f"  ! no prices for pitch {t} — skipped", file=sys.stderr)
            continue
        s = px[t].dropna()
        b = px[benchmark].dropna()
        on_or_after = s.index[s.index >= r["date"]]
        if len(on_or_after) == 0:
            print(f"  ! pitch date {r['date']} for {t} has no trading days yet — skipped",
                  file=sys.stderr)
            continue
        d0 = on_or_after[0]
        p0 = parse_money(r.get("price_at_pitch")) or float(s.loc[d0])
        ret = float(s.iloc[-1]) / p0 - 1
        bret = float(b.iloc[-1]) / float(b.loc[b.index[b.index >= r["date"]][0]]) - 1
        excess = ret - bret
        is_buy = (r.get("action") or "Buy").strip().lower() != "sell"
        vy, vn = parse_money(r.get("votes_yes")) or 0, parse_money(r.get("votes_no")) or 0
        pitches.append({
            "date": r["date"], "team": r.get("team") or "—", "ticker": t,
            "action": "Buy" if is_buy else "Sell",
            "votePct": 100 * vy / (vy + vn) if (vy + vn) else None,
            "accepted": (r.get("accepted") or "").strip().upper() == "Y",
            "vetoed": (r.get("vetoed") or "").strip().upper() == "Y",
            "priceAtPitch": round(p0, 2), "priceNow": round(float(s.iloc[-1]), 2),
            "ret": ret, "benchRet": bret, "excess": excess,
            "callScore": excess if is_buy else -excess,
            "notes": r.get("notes") or "",
        })
    return {"asOf": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "benchmark": benchmark, "pitches": pitches}


# ------------------------------------------------------------------- baking

DATA_DIR = os.path.join(HERE, "data")


def save_json(name, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, name), "w") as f:
        json.dump(data, f)


def bake_combined():
    """portfolio_dashboard.html carries both tabs (portfolio + macro).
    Either fetch script rebakes it from the latest saved JSON of each side,
    so the two refresh cadences stay independent."""
    def load(name):
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    with open(os.path.join(TEMPLATES, "portfolio_dashboard.html"), encoding="utf-8") as f:
        html = f.read()
    for ph, data in (("/*__DATA__*/null", load("portfolio.json")),
                     ("/*__MACRO__*/null", load("macro.json")),
                     ("/*__IMPACT__*/null", load("impact.json")),
                     ("/*__PITCHES__*/null", load("pitch_scoreboard.json"))):
        if data is not None:
            html = html.replace(
                ph, json.dumps(data, allow_nan=False).replace("</", "<\\/"), 1)
    os.makedirs(DIST, exist_ok=True)
    out = os.path.join(DIST, "portfolio_dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  wrote {os.path.relpath(out, HERE)} ({len(html)//1024} KB, combined)")


def bake(template_name, data):
    src = os.path.join(TEMPLATES, template_name)
    out = os.path.join(DIST, template_name)
    with open(src, encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, allow_nan=False, default=lambda o: None).replace("</", "<\\/")
    if "/*__DATA__*/null" not in html:
        raise RuntimeError(f"{template_name}: missing /*__DATA__*/null placeholder")
    html = html.replace("/*__DATA__*/null", blob, 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  wrote {os.path.relpath(out, HERE)} ({len(html)//1024} KB)")


def clean_nan(obj):
    """json.dumps(allow_nan=False) chokes on NaN floats from yfinance — scrub them."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", default=os.path.join(HERE, "holdings.csv"))
    ap.add_argument("--format", choices=["rjf", "ciq"], default="rjf")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--rf", type=float, default=4.0, help="risk-free rate, annual %%")
    ap.add_argument("--comps", default="AAPL:MSFT,GOOGL,META,NVDA",
                    help="SUBJECT:PEER1,PEER2,...")
    ap.add_argument("--dcf", default="AAPL,MSFT,NVDA,KO,NKE",
                    help="tickers to bake as DCF presets")
    ap.add_argument("--skip-news", action="store_true",
                    help="skip per-stock headline fetch (much faster; keeps cached)")
    ap.add_argument("--pitches", default=os.path.join(HERE, "pitches.csv"),
                    help="pitch log CSV exported from pitch_tracker.html")
    args = ap.parse_args()

    os.makedirs(DIST, exist_ok=True)
    cache = load_cache()

    parser = parse_rjf if args.format == "rjf" else parse_ciq
    holdings = parser(args.holdings)
    print(f"Parsed {len(holdings)} equity positions from {os.path.basename(args.holdings)}")

    benches = list(BENCHMARKS)
    if args.benchmark not in [b[0] for b in benches]:
        benches.insert(0, (args.benchmark, args.benchmark))
    else:  # requested benchmark becomes the default (first) entry
        benches.sort(key=lambda b: b[0] != args.benchmark)

    live, port = build_portfolio_data(holdings, benches, args.rf)
    add_sectors(port, live, cache)
    enrich_details(port, cache, fetch_news=not args.skip_news)
    daily = port.pop("_daily")
    spy_daily = port.pop("_spyDaily")
    port["factors"] = build_factor_models(daily, spy_daily)
    if port["factors"] and port["factors"].get("sharpe") is not None:
        # override the flat-rf Sharpe so the whole scorecard uses the daily T-bill
        port["sharpe"] = port["factors"]["sharpe"]
    add_risk(port, daily, cache)
    if os.path.exists(args.pitches):
        port["pitchMarkers"] = [
            {"date": r["date"], "ticker": (r.get("ticker") or "").strip().upper(),
             "team": r.get("team") or "",
             "action": (r.get("action") or "Buy").strip(),
             "accepted": (r.get("accepted") or "").strip().upper() == "Y",
             "vetoed": (r.get("vetoed") or "").strip().upper() == "Y"}
            for r in parse_pitches(args.pitches) if r.get("date")]
    # SAFETY: a throttled/resource-starved fetch can drop most holdings
    # (Yahoo returns empty columns). Never overwrite good portfolio data —
    # and never publish — with a gutted run. Mirrors the guard in
    # shard_universe.py.
    prev_path = os.path.join(DATA_DIR, "portfolio.json")
    if os.path.exists(prev_path):
        try:
            prev_n = len(json.load(open(prev_path)).get("positions", []))
        except Exception:
            prev_n = 0
        now_n = len(port["positions"])
        if prev_n and now_n < 0.75 * prev_n:
            sys.exit(f"REFUSING to overwrite portfolio.json: only {now_n} positions "
                     f"this run vs {prev_n} previously ({len(port.get('excluded', []))} "
                     f"excluded). Keeping existing data — investigate the price fetch.")

    save_json("portfolio.json", clean_nan(port))
    bake_combined()

    subject, peers = args.comps.split(":")
    comps = build_comps_data(subject.strip(), [p.strip() for p in peers.split(",") if p.strip()], cache)
    bake("comps.html", clean_nan(comps))

    presets = build_dcf_presets([t.strip() for t in args.dcf.split(",") if t.strip()], cache)
    bake("dcf_sandbox.html", clean_nan({"presets": presets}))

    if os.path.exists(args.pitches):
        rows = parse_pitches(args.pitches)
        if rows:
            save_json("pitch_scoreboard.json",
                      clean_nan(build_pitch_data(rows, args.benchmark)))
            bake_combined()
    else:
        print(f"No {os.path.basename(args.pitches)} — skipping pitch scoreboard.")

    save_cache(cache)
    print("Done. Open the files in dist/ — they are self-contained.")


if __name__ == "__main__":
    main()
