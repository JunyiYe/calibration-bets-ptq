from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    if len(yt) < 2 or np.ptp(yt) < 1e-12 or np.ptp(yp) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(yt, yp)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def rank_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_rank = pd.Series(y_true.reshape(-1)).rank().to_numpy()
    pred_rank = pd.Series(y_pred.reshape(-1)).rank().to_numpy()
    return information_coefficient(true_rank, pred_rank)


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) == 0 or np.std(returns) < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(returns) / np.std(returns))


def max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    safe_returns = np.clip(returns, -0.99, 10.0)
    equity = np.cumprod(1.0 + safe_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def summarize_finance(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ic": information_coefficient(y_true, y_pred),
        "rank_ic": rank_information_coefficient(y_true, y_pred),
    }
