from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def daily_ic_frame(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    min_assets: int = 5,
    rank: bool = False,
) -> pd.DataFrame:
    """Cross-sectional IC per trading day for one or more prediction series.

    Returns a DataFrame indexed by date (chronological) with one IC column per
    entry in `preds`. Days with fewer than `min_assets` observations, a
    degenerate target cross-section, or a degenerate prediction cross-section
    for ANY series are dropped for all series alike, so columns are aligned
    and differences between them are paired by day.
    """
    base = pd.DataFrame({"date": meta["date"].to_numpy(), "y_true": np.asarray(y_true).reshape(-1)})
    for name, p in preds.items():
        if name == "y_true":
            raise ValueError("Prediction series may not be named 'y_true'.")
        base[name] = np.asarray(p).reshape(-1)

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for day, g in base.groupby("date", sort=True):
        if len(g) < min_assets or float(g["y_true"].std()) < 1e-8:
            continue
        yt = g["y_true"].rank() if rank else g["y_true"]
        row = {}
        for name in preds:
            yp = g[name].rank() if rank else g[name]
            row[name] = float(yt.corr(yp)) if float(yp.std()) > 1e-12 else np.nan
        rows[day] = row
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return frame.dropna()


def newey_west_mean_test(series: np.ndarray, lag: int) -> tuple[float, float, float, float]:
    """Mean-zero test with Newey-West HAC standard error.

    When targets come from overlapping forward windows (forward_period > 1),
    consecutive daily values are serially correlated, so the naive i.i.d.-days
    SE understates the true standard error. Use lag = forward_period - 1.
    Returns (mean, tstat, pvalue, se).
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    if n < 2:
        return float(series.mean()) if n else 0.0, 0.0, 1.0, 0.0
    xbar = float(series.mean())
    resid = series - xbar
    var = float(np.mean(resid**2))
    for l in range(1, min(lag, n - 1) + 1):
        gamma_l = float(np.mean(resid[l:] * resid[:-l]))
        var += 2.0 * (1.0 - l / (lag + 1)) * gamma_l
    var = max(var, 0.0)
    se = float(np.sqrt(var / n))
    tstat = xbar / se if se > 0 else 0.0
    pval = float(2 * stats.t.sf(abs(tstat), df=max(n - 1, 1)))
    return xbar, tstat, pval, se


def paired_delta_tests(
    frame: pd.DataFrame,
    baseline: str,
    lag: int,
) -> dict[str, dict[str, float]]:
    """Paired-by-day tests of IC degradation vs a baseline column.

    delta = IC_baseline - IC_variant per day, so positive mean = degradation.
    Returns per-variant {ic_mean, delta_ic_mean, tstat, pvalue, se, n_days}.
    """
    if baseline not in frame.columns:
        raise ValueError(f"Baseline column {baseline!r} not in frame.")
    out: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        if col == baseline:
            continue
        delta = (frame[baseline] - frame[col]).to_numpy()
        mean, tstat, pval, se = newey_west_mean_test(delta, lag)
        out[col] = {
            "ic_mean": float(frame[col].mean()),
            "delta_ic_mean": mean,
            "tstat": tstat,
            "pvalue": pval,
            "se": se,
            "n_days": float(len(delta)),
        }
    return out
