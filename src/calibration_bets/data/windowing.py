from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowConfig:
    context_length: int
    horizon: int = 1
    forward_period: int = 1
    normalize: str = "zscore"
    target_transform: str = "raw"
    target_clip: tuple[float, float] | None = None
    feature_columns: tuple[str, ...] = ("value",)
    target_agg: str = "sum"  # "sum" (forward return) | "realized_vol" (std of forward returns)
    target_log: bool = False  # log(x + eps) before cross-sectional z-scoring; for skewed targets (e.g. volatility)


class TimeSeriesWindowDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, df: pd.DataFrame, config: WindowConfig) -> None:
        self.config = config
        self.x, self.y, self.raw_y, self.meta = make_windows(df, config)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx]).float()

    @property
    def num_features(self) -> int:
        return self.x.shape[-1]

    @classmethod
    def from_arrays(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        raw_y: np.ndarray,
        meta: pd.DataFrame,
        config: WindowConfig,
    ) -> "TimeSeriesWindowDataset":
        obj = cls.__new__(cls)
        obj.config = config
        obj.x, obj.y, obj.raw_y = x, y, raw_y
        obj.meta = meta.reset_index(drop=True)
        return obj

    def subset_by_date(
        self,
        start_exclusive: object = None,
        end_inclusive: object = None,
    ) -> "TimeSeriesWindowDataset":
        """Windows whose TARGET date falls in (start_exclusive, end_inclusive].

        Assigning by target date lets a window's context span backward across a
        split boundary — backward-looking data only, so no leakage — instead of
        discarding the first context_length+forward_period days of each split.
        """
        dates = self.meta["date"]
        mask = np.ones(len(dates), dtype=bool)
        if start_exclusive is not None:
            mask &= (dates > pd.Timestamp(start_exclusive)).to_numpy()
        if end_inclusive is not None:
            mask &= (dates <= pd.Timestamp(end_inclusive)).to_numpy()
        return type(self).from_arrays(
            self.x[mask], self.y[mask], self.raw_y[mask], self.meta.loc[mask], self.config
        )


def make_windows(
    df: pd.DataFrame,
    config: WindowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metas: list[dict[str, object]] = []

    feat_cols = list(config.feature_columns)
    target_col = feat_cols[0]  # always use the first feature column as the target

    for asset, g in df.sort_values(["asset", "date"]).groupby("asset", sort=False):
        arr = g[feat_cols].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        n = len(arr)
        fwd = config.forward_period
        last_start = n - config.context_length - fwd + 1
        for start in range(max(0, last_start)):
            end = start + config.context_length
            target_idx = end + fwd - 1
            x = arr[start:end].copy()  # (context_length, num_features)
            fwd_vals = arr[end : end + fwd, 0].astype(np.float32)
            if config.target_agg == "sum":
                y = fwd_vals.sum(keepdims=True)  # sum of N-day forward returns
            elif config.target_agg == "realized_vol":
                y = np.array([fwd_vals.std()], dtype=np.float32)  # realized vol over N-day forward window
            else:
                raise ValueError(f"Unknown target_agg: {config.target_agg}")
            if config.normalize == "zscore":
                mean = x.mean(axis=0, keepdims=True)
                std = x.std(axis=0, keepdims=True)
                std = np.where(std > 1e-8, std, 1.0)
                x = (x - mean) / std
            xs.append(x)
            ys.append(y)
            metas.append({"asset": asset, "date": pd.Timestamp(dates[target_idx])})

    if not xs:
        raise ValueError("No windows were created. Check context_length, horizon, and data length.")
    raw_y = np.stack(ys)
    meta = pd.DataFrame(metas)
    target_y = transform_targets(raw_y, meta, config.target_transform, config.target_clip, config.target_log)
    return np.stack(xs), target_y, raw_y, meta


def transform_targets(
    y: np.ndarray,
    meta: pd.DataFrame,
    method: str,
    clip_range: tuple[float, float] | None = None,
    log_transform: bool = False,
) -> np.ndarray:
    if method in {"raw", "none"}:
        out = y.copy()
        if log_transform:
            out = np.log(out + 1e-6).astype(np.float32)
        if clip_range is not None:
            out = np.clip(out, clip_range[0], clip_range[1])
        return out
    if method != "cross_sectional_zscore":
        raise ValueError(f"Unknown target_transform: {method}")
    if y.shape[1] != 1:
        raise ValueError("cross_sectional_zscore target_transform currently supports horizon=1 only.")

    target = y.reshape(-1).astype(np.float32)
    if log_transform:
        target = np.log(target + 1e-6).astype(np.float32)
    out = np.zeros_like(target)
    dates = meta["date"].to_numpy()
    for date in pd.unique(dates):
        mask = dates == date
        values = target[mask]
        std = float(values.std())
        if std > 1e-8:
            out[mask] = (values - float(values.mean())) / std
        else:
            out[mask] = 0.0
    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])
    return out.reshape(y.shape).astype(np.float32)
