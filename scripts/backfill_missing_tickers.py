"""Backfill deep history for tickers that failed during the panel download.

Yahoo intermittently serves empty data for individual symbols (and 429s the
whole IP after a bulk download). This script retries them, validates the
fresh series against the rows already in the panel on their overlap, and
replaces those rows with the full-depth series when the check passes.

    python scripts/backfill_missing_tickers.py \
        --panel data/sp500_returns.csv --tickers BK,CTRA,DAY,HOLX,MMC
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import yfinance as yf


def seed_tz_cache(tickers: list[str]) -> None:
    # Skip the flaky quoteSummary timezone lookup for known-US symbols.
    try:
        from yfinance import cache as yfcache

        tzc = yfcache._TzCacheManager.get_tz_cache()
        for s in tickers:
            tzc.store(s, "America/New_York")
    except Exception as exc:  # noqa: BLE001 - cache layout varies by version
        print(f"[WARN] tz cache seeding unavailable: {exc}")


def fetch_returns(ticker: str, start: str, end: str, retries: int) -> pd.Series | None:
    for attempt in range(1, retries + 1):
        try:
            h = yf.Ticker(ticker.replace(".", "-")).history(start=start, end=end, auto_adjust=True)
            if len(h) > 100:
                close = h["Close"].dropna()
                close.index = pd.to_datetime(close.index).tz_localize(None)
                r = close.pct_change().dropna()
                return r[(r.index >= start) & (r.index < end)]
            print(f"[thin] {ticker}: {len(h)} rows (attempt {attempt})")
        except Exception as exc:  # noqa: BLE001
            print(f"[retry] {ticker}: {str(exc)[:80]}")
        time.sleep(2.0 * attempt)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed tickers and merge deep history into the panel.")
    parser.add_argument("--panel", default="data/sp500_returns.csv")
    parser.add_argument("--tickers", default="BK,CTRA,DAY,HOLX,MMC")
    parser.add_argument("--start", default="2008-06-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-overlap-diff", type=float, default=1e-3,
                        help="Reject a fresh series if it disagrees with existing panel rows beyond this.")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    seed_tz_cache(tickers)
    panel = pd.read_csv(args.panel, parse_dates=["date"])

    replaced = []
    for t in tickers:
        fresh = fetch_returns(t, args.start, args.end, args.retries)
        if fresh is None:
            print(f"[FAIL] {t}: still unavailable, panel rows kept as-is")
            continue
        existing = panel[panel.symbol == t].set_index("date")["return"]
        overlap = pd.concat([existing, fresh], axis=1, join="inner")
        if len(overlap):
            diff = (overlap.iloc[:, 0] - overlap.iloc[:, 1]).abs().max()
            if diff > args.max_overlap_diff:
                print(f"[REJECT] {t}: overlap max|diff|={diff:.5f} > {args.max_overlap_diff}")
                continue
            print(f"[OK] {t}: {len(fresh)} rows ({fresh.index.min().date()}..), "
                  f"overlap {len(overlap)}d max|diff|={diff:.2e}")
        panel = panel[panel.symbol != t]
        panel = pd.concat([panel, pd.DataFrame({"date": fresh.index, "symbol": t, "return": fresh.values})])
        replaced.append(t)

    if replaced:
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
        panel.to_csv(args.panel, index=False)
        print(f"[OK] wrote {args.panel}: rows={len(panel)}, symbols={panel.symbol.nunique()}, replaced={replaced}")
    else:
        print("[NOOP] nothing replaced; panel unchanged")


if __name__ == "__main__":
    main()
