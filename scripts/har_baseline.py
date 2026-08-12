"""HAR-RV baseline for the volatility task — the classic econometric reference.

Pooled log-HAR: log(fwd 5d realized vol) ~ 1 + log(rv_1d) + log(rv_5d) + log(rv_22d),
fit on each fold's training dates, evaluated as mean daily cross-sectional IC
on the fold's test year (same walk-forward folds as the deep models). Also
reports the no-fit persistence baseline (predict past 5d vol).

    python scripts/har_baseline.py --data data/sp500_returns.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibration_bets.data.panel import PanelSpec, load_csv_panel
from calibration_bets.metrics.significance import daily_ic_frame, newey_west_mean_test

EPS = 1e-6


def build_har_table(panel: pd.DataFrame, fwd: int = 5) -> pd.DataFrame:
    rows = []
    for asset, g in panel.sort_values(["asset", "date"]).groupby("asset", sort=False):
        r = g["value"].reset_index(drop=True)
        y = r.rolling(fwd).std()                       # at j: std of j-4..j
        rv1 = r.abs().shift(fwd)                        # info date i = j - fwd
        rv5 = r.rolling(5).std().shift(fwd)
        rv22 = r.rolling(22).std().shift(fwd)
        rows.append(pd.DataFrame({
            "date": g["date"].to_numpy(), "asset": asset,
            "y": np.log(y + EPS),
            "x1": np.log(rv1 + EPS), "x5": np.log(rv5 + EPS), "x22": np.log(rv22 + EPS),
            "persistence": np.log(rv5 + EPS),
        }))
    return pd.concat(rows, ignore_index=True).dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="HAR-RV baseline on walk-forward folds.")
    parser.add_argument("--data", default="data/sp500_returns.csv")
    parser.add_argument("--folds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--fwd", type=int, default=5)
    parser.add_argument("--min-history", type=int, default=1000)
    parser.add_argument("--train-years", type=int, default=None,
                        help="Fixed training-window length in years (default: expanding).")
    args = parser.parse_args()

    panel = load_csv_panel(PanelSpec(path=args.data, min_history=args.min_history))
    table = build_har_table(panel, fwd=args.fwd)
    hac_lag = args.fwd - 1

    report = {"data": args.data, "folds": {}}
    for fold in [int(f) for f in str(args.folds).split(",") if f]:
        train_end = pd.Timestamp(f"{fold - 2}-12-31")
        test_lo, test_hi = pd.Timestamp(f"{fold - 1}-12-31"), pd.Timestamp(f"{fold}-12-31")
        # walk-forward parity with the deep models: fit through Y-2, skip the
        # models' validation year, test on year Y
        train = table[table["date"] <= train_end]
        if args.train_years is not None:
            train = train[train["date"] > pd.Timestamp(f"{fold - 2 - args.train_years}-12-31")]
        test = table[(table["date"] > test_lo) & (table["date"] <= test_hi)]
        X = np.column_stack([np.ones(len(train)), train[["x1", "x5", "x22"]].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)
        Xt = np.column_stack([np.ones(len(test)), test[["x1", "x5", "x22"]].to_numpy()])
        har_pred = Xt @ beta

        frame = daily_ic_frame(
            test[["date"]], test["y"].to_numpy(),
            {"har": har_pred, "persistence": test["persistence"].to_numpy()},
        )
        fold_out = {"n_days": int(len(frame)), "beta": [float(b) for b in beta]}
        for k in ("har", "persistence"):
            mean, tstat, pval, _ = newey_west_mean_test(frame[k].to_numpy(), lag=hac_lag)
            fold_out[k] = {"daily_ic_mean": mean, "tstat": tstat, "pvalue": pval}
        report["folds"][str(fold)] = fold_out
        print(f"fold {fold}: HAR IC={fold_out['har']['daily_ic_mean']:+.4f} (t={fold_out['har']['tstat']:.1f})  "
              f"persistence IC={fold_out['persistence']['daily_ic_mean']:+.4f}  n_days={fold_out['n_days']}")

    for k in ("har", "persistence"):
        ics = [report["folds"][f][k]["daily_ic_mean"] for f in report["folds"]]
        report[f"{k}_mean_ic"] = float(np.mean(ics))
    out = Path("outputs") / "har_baseline.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"HAR mean IC across folds: {report['har_mean_ic']:+.4f}   "
          f"persistence: {report['persistence_mean_ic']:+.4f}")
    print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    main()
