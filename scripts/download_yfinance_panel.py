from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def main() -> None:
    parser = argparse.ArgumentParser(description="Download adjusted-close returns from yfinance.")
    parser.add_argument("--tickers", default="configs/sp500_tickers.txt")
    parser.add_argument("--output", default="data/sp500_returns.csv")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between tickers (rate-limit courtesy).")
    args = parser.parse_args()

    cache_dir = Path("outputs/yfinance_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

    tickers = read_tickers(args.tickers)
    series: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}

    for ticker in tickers:
        yf_ticker = ticker.replace(".", "-")
        try:
            series[ticker] = download_close_with_retry(
                yf_ticker,
                start=args.start,
                end=args.end,
                retries=args.retries,
            )
            print(f"[OK] {ticker}: {len(series[ticker])} rows")
        except Exception as exc:  # noqa: BLE001 - keep downloader resilient.
            failures[ticker] = repr(exc)
            print(f"[WARN] {ticker} failed: {exc}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    if not series:
        raise RuntimeError(f"No ticker data downloaded. Failures: {failures}")

    prices = pd.DataFrame(series).sort_index()
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    panel = returns.stack().dropna().reset_index()
    panel.columns = ["date", "symbol", "return"]
    panel = panel.sort_values(["date", "symbol"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    print(f"[OK] wrote {out} rows={len(panel)} symbols={panel['symbol'].nunique()}")
    print(f"[OK] date range: {panel['date'].min()} -> {panel['date'].max()}")
    if failures:
        print(f"[WARN] failed symbols: {failures}")


def read_tickers(path: str) -> list[str]:
    tickers = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.append(line)
    if not tickers:
        raise ValueError(f"No tickers found in {path}")
    return tickers


def download_close_with_retry(ticker: str, start: str, end: str, retries: int) -> pd.Series:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            close = extract_close(data, ticker)
            if close.dropna().empty:
                raise RuntimeError("downloaded close series is empty")
            return close.dropna()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last_exc}")


def extract_close(data: pd.DataFrame, ticker: str) -> pd.Series:
    if data.empty:
        raise RuntimeError("empty dataframe")
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                return close.iloc[:, 0].rename(ticker)
            return close.rename(ticker)
    if "Close" in data.columns:
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0].rename(ticker)
        return close.rename(ticker)
    raise ValueError(f"Could not find Close in columns: {data.columns}")


if __name__ == "__main__":
    main()
