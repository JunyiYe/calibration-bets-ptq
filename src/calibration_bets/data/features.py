from __future__ import annotations

import numpy as np
import pandas as pd

DERIVED_FEATURES = (
    "ret_ma5", "ret_ma10", "ret_ma20", "vol_20", "rsi_14",
    "macd", "macd_signal", "max_20", "min_20", "skew_20", "mom_12m",
)


def add_return_features(
    df: pd.DataFrame,
    asset_col: str = "asset",
    value_col: str = "value",
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Add backward-looking derived features per asset.

    Only the features named in `feature_columns` are computed, and only their
    warmup NaN rows are dropped. Computing everything unconditionally would
    make every config pay the longest warmup — mom_12m alone needs ~273 rows
    per asset — even when the feature is unused. `feature_columns=None`
    computes the full set.
    """
    if feature_columns is None:
        needed = set(DERIVED_FEATURES)
    else:
        needed = set(feature_columns) - {value_col, "value"}
        unknown = needed - set(DERIVED_FEATURES)
        if unknown:
            raise ValueError(f"Unknown derived features requested: {sorted(unknown)}")
    if not needed:
        return df

    df = df.sort_values([asset_col, "date"]).copy()
    for asset in df[asset_col].unique():
        mask = df[asset_col] == asset
        s = df.loc[mask, value_col]
        # rolling mean returns (multi-window momentum)
        if "ret_ma5" in needed:
            df.loc[mask, "ret_ma5"] = s.rolling(5, min_periods=1).mean()
        if "ret_ma10" in needed:
            df.loc[mask, "ret_ma10"] = s.rolling(10, min_periods=1).mean()
        if "ret_ma20" in needed:
            df.loc[mask, "ret_ma20"] = s.rolling(20, min_periods=1).mean()
        # rolling volatility
        if "vol_20" in needed:
            df.loc[mask, "vol_20"] = s.rolling(20, min_periods=2).std()
        # RSI (14-day)
        if "rsi_14" in needed:
            df.loc[mask, "rsi_14"] = _rsi(s, 14)
        # MACD (12-26 EMA difference) + signal line (9-day EMA of MACD)
        if needed & {"macd", "macd_signal"}:
            ema_12 = s.ewm(span=12, min_periods=1).mean()
            ema_26 = s.ewm(span=26, min_periods=1).mean()
            macd = ema_12 - ema_26
            if "macd" in needed:
                df.loc[mask, "macd"] = macd
            if "macd_signal" in needed:
                df.loc[mask, "macd_signal"] = macd.ewm(span=9, min_periods=1).mean()
        # rolling extremes (20-day)
        if "max_20" in needed:
            df.loc[mask, "max_20"] = s.rolling(20, min_periods=1).max()
        if "min_20" in needed:
            df.loc[mask, "min_20"] = s.rolling(20, min_periods=1).min()
        # rolling skewness (20-day)
        if "skew_20" in needed:
            df.loc[mask, "skew_20"] = s.rolling(20, min_periods=3).skew()
        # 12-month momentum, skipping the most recent month (standard momentum
        # construction — avoids contamination from short-term reversal)
        if "mom_12m" in needed:
            df.loc[mask, "mom_12m"] = s.rolling(252, min_periods=252).sum().shift(21)
    # drop rows with NaN features from warmup periods (first few rows per asset)
    df = df.dropna(subset=sorted(needed)).reset_index(drop=True)
    return df


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_panel_stats(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, tuple[float, float]]:
    stats = {}
    for col in feature_columns:
        vals = df[col].dropna().values
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-12:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def normalize_panel(
    df: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    feature_columns: list[str],
    clip_range: tuple[float, float] | None = (-3, 3),
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_columns:
        mean, std = stats[col]
        df[col] = (df[col].astype(np.float64) - mean) / std
    if clip_range is not None:
        low, high = clip_range
        for col in feature_columns:
            if col == "value":
                continue  # don't clip value — it's used as the target source
            df[col] = df[col].clip(low, high)
    return df


DEFAULT_FEATURE_COLUMNS = [
    "value", "ret_ma5", "ret_ma10", "ret_ma20", "vol_20", "rsi_14",
    "macd", "macd_signal", "max_20", "min_20", "skew_20",
]
