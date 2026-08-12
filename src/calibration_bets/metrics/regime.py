from __future__ import annotations

import numpy as np
import pandas as pd

from calibration_bets.metrics.finance import summarize_finance
from calibration_bets.metrics.forecast import summarize_forecast


def summarize_by_abs_return_decile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_deciles: int = 10,
    forecast_true: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate metrics by absolute-return decile.

    This is a lightweight proxy for volatility/tail regimes. It is intentionally
    based only on realized targets, so it works for public CSV panels before we
    add external regime labels such as VIX or realized-vol windows.
    """
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ft = yt if forecast_true is None else forecast_true.reshape(-1)
    if len(yt) != len(yp) or len(ft) != len(yp):
        raise ValueError("y_true, forecast_true, and y_pred must have the same flattened length.")
    if len(yt) == 0:
        return {}

    q = min(n_deciles, len(np.unique(np.abs(yt))))
    if q < 2:
        return {"all": _summarize_bucket(yt, yp, ft)}

    buckets = pd.qcut(np.abs(yt), q=q, labels=False, duplicates="drop")
    result: dict[str, dict[str, float]] = {}
    for bucket in sorted(pd.Series(buckets).dropna().unique()):
        mask = np.asarray(buckets == bucket)
        label = f"abs_return_decile_{int(bucket)}"
        result[label] = _summarize_bucket(yt[mask], yp[mask], ft[mask])
    return result


def _summarize_bucket(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    forecast_true: np.ndarray,
) -> dict[str, float]:
    metrics = {"count": float(len(y_true))}
    metrics.update(summarize_forecast(forecast_true, y_pred))
    metrics.update(summarize_finance(y_true, y_pred))
    return metrics
