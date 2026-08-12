from __future__ import annotations

import numpy as np
import pandas as pd

from calibration_bets.metrics.finance import max_drawdown, sharpe_ratio


def long_short_backtest(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_quantile: float = 0.1,
    bottom_quantile: float = 0.1,
    transaction_cost_bps: float = 5.0,
) -> dict[str, float]:
    df = meta.copy()
    df["actual"] = y_true.reshape(-1)
    df["pred"] = y_pred.reshape(-1)

    daily_returns: list[float] = []
    daily_turnover: list[float] = []
    prev_weights: pd.Series | None = None
    cost = transaction_cost_bps / 10_000.0

    for _, day in df.groupby("date", sort=True):
        if len(day) < 4:
            continue
        n_long = max(1, int(np.floor(len(day) * top_quantile)))
        n_short = max(1, int(np.floor(len(day) * bottom_quantile)))
        if n_long + n_short > len(day):
            continue

        ordered = day.sort_values(["pred", "asset"], ascending=[False, True])
        long_assets = ordered.head(n_long)["asset"]
        short_assets = ordered.tail(n_short)["asset"]
        overlap = set(long_assets).intersection(set(short_assets))
        if overlap:
            continue

        weights = pd.Series(0.0, index=day["asset"].to_numpy())
        weights.loc[long_assets.to_numpy()] = 1.0 / len(long_assets)
        weights.loc[short_assets.to_numpy()] = -1.0 / len(short_assets)

        actual = day.set_index("asset")["actual"]
        gross_return = float((weights * actual.reindex(weights.index)).sum())
        if prev_weights is None:
            turnover = float(weights.abs().sum())
        else:
            all_assets = weights.index.union(prev_weights.index)
            aligned_cur = weights.reindex(all_assets).fillna(0.0)
            aligned_prev = prev_weights.reindex(all_assets).fillna(0.0)
            turnover = float((aligned_cur - aligned_prev).abs().sum())

        daily_returns.append(gross_return - cost * turnover)
        daily_turnover.append(turnover)
        prev_weights = weights

    returns = np.array(daily_returns, dtype=np.float64)
    return {
        "portfolio_return_mean": float(returns.mean()) if len(returns) else 0.0,
        "portfolio_return_sum": float(returns.sum()) if len(returns) else 0.0,
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "turnover_mean": float(np.mean(daily_turnover)) if daily_turnover else 0.0,
        "num_backtest_days": float(len(returns)),
    }
