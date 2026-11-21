#!/usr/bin/env python3
"""
FIN 233 live-lookup server for the Pitch impact tab.

Run this on your machine and the dashboard's "Search any NYSE/Nasdaq stock"
box comes alive: type a ticker (or company name), and the page fetches the
full pitch-impact payload — 3y impact series vs the class portfolio, analyst
detail, headlines, earnings + Polymarket odds — computed on demand.

    python3 impact_server.py [--port 8901]

Without this server running, the dashboard quietly falls back to the baked
candidate dropdown (which is all a shared/emailed copy ever sees).

Stdlib + the existing fetch helpers; CORS is open so the static page
(file:// or localhost) can call it.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import yfinance as yf

from fetch_data import (HERE, SYMBOL_FIXES, clean_nan, load_cache,
                        load_news_cache, parse_rjf, save_cache)
from pitch_impact import BASE_PATH, build_candidate, make_base

SYMBOLS_PATH = os.path.join(HERE, "data", "symbols.json")
SYMBOLS_TTL_DAYS = 7
BASE_MAX_AGE_DAYS = 3

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
NYSE_CODES = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca"}

LOCK = threading.RLock()
STATE = {"base": None, "symbols": [], "pm": None, "pm_ts": 0,
         "cache": None, "news_cache": None, "args": None}


# ------------------------------------------------------------- symbol list

def fetch_symbol_directory():
    if os.path.exists(SYMBOLS_PATH):
        try:
            blob = json.load(open(SYMBOLS_PATH))
            if time.time() - blob.get("_ts", 0) < SYMBOLS_TTL_DAYS * 86400:
                return blob["symbols"]
        except Exception:
            pass
    print("Downloading NASDAQ/NYSE symbol directory ...")
    out = []
    for url, kind in ((NASDAQ_URL, "nasdaq"), (OTHER_URL, "other")):
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.4.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            lines = r.read().decode("latin-1").splitlines()
        hdr = lines[0].split("|")
        for line in lines[1:]:
            p = line.split("|")
            if len(p) != len(hdr) or line.startswith("File Creation"):
                continue
            row = dict(zip(hdr, p))
            if row.get("Test Issue") == "Y":
                continue
            if kind == "nasdaq":
                out.append([row["Symbol"], row["Security Name"], "NASDAQ"])
            else:
                exch = NYSE_CODES.get(row.get("Exchange"))
                if exch:
                    out.append([row["ACT Symbol"], row["Security Name"], exch])
    out.sort(key=lambda s: s[0])
    os.makedirs(os.path.dirname(SYMBOLS_PATH), exist_ok=True)
    json.dump({"_ts": time.time(), "symbols": out}, open(SYMBOLS_PATH, "w"))
    print(f"  {len(out)} listed symbols cached")
    return out


def yahoo_symbol(sym):
    """Directory notation -> Yahoo (BRK.B -> BRK-B), plus explicit fixes."""
    sym = sym.strip().upper()
    return SYMBOL_FIXES.get(sym, sym.replace(".", "-").replace("$", "-P"))


# ------------------------------------------------------------------ base

def load_or_build_base(args):
    base = None
    if os.path.exists(BASE_PATH):
        try:
            base = json.load(open(BASE_PATH))
            built = datetime.strptime(base["built"], "%Y-%m-%d %H:%M")
            age = (datetime.now() - built).days
            if age <= BASE_MAX_AGE_DAYS:
                print(f"Loaded portfolio base (built {base['built']}, "
                      f"{len(base['portRet'])} days)")
                return base
            print(f"Portfolio base is {age}d old — rebuilding ...")
        except Exception as e:
            print(f"Could not read base ({e}) — rebuilding ...")
    if not os.path.exists(args.holdings):
        # cloud deployment: no holdings.csv on the box — the base ships in the
        # repo and is refreshed by the nightly push, so use it as-is
        if base is not None:
            print("No holdings.csv here — using the shipped base as-is "
                  f"(built {base.get('built')})")
            return base
        sys.exit("No portfolio base and no holdings.csv — cannot start.")
    holdings = parse_rjf(args.holdings)
    base = make_base(holdings, args.years, args.benchmark, args.rf, STATE["cache"])
    save_cache(STATE["cache"])
    return base


def get_pm():
    """Polymarket earnings map, cached 30 min. Keyed by ticker."""
    with LOCK:
        if STATE["pm"] is not None and time.time() - STATE["pm_ts"] < 1800:
            return STATE["pm"]
    try:
        from fetch_macro import build_polymarket
        pm = build_polymarket(None)  # None -> match everything (see fetch_macro)
    except Exception as e:
        print(f"  ! Polymarket lookup skipped: {e}", file=sys.stderr)
        pm = {}
    with LOCK:
        STATE["pm"], STATE["pm_ts"] = pm, time.time()
    return pm


# ---------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/health":
            b = STATE["base"]
            return self.send_json(200, {
                "ok": True, "built": b["built"], "benchmark": b["benchmark"],
                "rf": b["rf"], "years": b["years"], "symbols": len(STATE["symbols"])})
        if u.path == "/api/symbols":
            return self.send_json(200, STATE["symbols"])
        if u.path == "/api/candidate":
            q = parse_qs(u.query)
            raw = (q.get("t") or [""])[0].strip().upper()
            if not re.fullmatch(r"[A-Z0-9.$-]{1,10}", raw or ""):
                return self.send_json(400, {"error": "bad ticker"})
            sym = yahoo_symbol(raw)
            try:
                cpx = yf.download(sym, period=f"{STATE['base']['years']}y",
                                  interval="1d", auto_adjust=True,
                                  progress=False)["Close"]
                cpx = cpx[sym] if hasattr(cpx, "columns") and sym in cpx.columns else cpx.squeeze()
                cpx = cpx.dropna()
            except Exception as e:
                return self.send_json(502, {"error": f"price fetch failed: {e}"})
            if cpx.empty:
                return self.send_json(404, {"error": f"no price data for {raw}"})
            pm = get_pm()  # fetched outside the state lock (network call)
            with LOCK:
                cand = build_candidate(sym, cpx, STATE["base"], STATE["cache"],
                                       STATE["news_cache"], pm)
                save_cache(STATE["cache"])
                json.dump(STATE["news_cache"],
                          open(os.path.join(HERE, ".news_cache.json"), "w"))
            if not cand:
                return self.send_json(422, {"error":
                    f"{raw}: not enough overlapping trading history (needs ~6 months)"})
            print(f"  served {sym} ({cand['windowDays']} days)")
            return self.send_json(200, clean_nan(cand))
        self.send_json(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (cloud deploys use 0.0.0.0)")
    ap.add_argument("--holdings", default=os.path.join(HERE, "holdings.csv"))
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--rf", type=float, default=4.0)
    args = ap.parse_args()

    STATE["args"] = args
    STATE["cache"] = load_cache()
    STATE["news_cache"] = load_news_cache()
    STATE["base"] = load_or_build_base(args)
    STATE["symbols"] = fetch_symbol_directory()

    print("=" * 56)
    print("FIN 233 impact lookup server")
    print(f"  http://127.0.0.1:{args.port}   (the dashboard finds it automatically)")
    print(f"  {len(STATE['symbols'])} NASDAQ/NYSE symbols · portfolio base {STATE['base']['built']}")
    print("=" * 56)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
