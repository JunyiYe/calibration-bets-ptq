from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationConfig:
    policy: str
    num_windows: int
    volatility_window: int = 20
    seed: int = 7


def select_calibration_indices(
    meta: pd.DataFrame,
    targets: np.ndarray,
    config: CalibrationConfig,
    inputs: np.ndarray | None = None,
) -> np.ndarray:
    """Select window indices for calibration-policy ablations.

    Dynamic INT8 does not need a calibration set, but this interface is used by
    static/fake-quant and by the paper's calibration-policy analysis.
    """
    n = len(meta)
    if n == 0:
        raise ValueError("Cannot select calibration indices from an empty set.")
    k = min(config.num_windows, n)
    rng = np.random.default_rng(config.seed)

    if config.policy == "random":
        return np.sort(rng.choice(n, size=k, replace=False))

    if config.policy == "recent":
        order = np.argsort(meta["date"].to_numpy())
        return np.sort(order[-k:])

    if config.policy == "volatility_stratified":
        vol = input_window_volatility(inputs, config.volatility_window)
        buckets = pd.qcut(vol, q=min(10, n), labels=False, duplicates="drop")
        bucket_series = pd.Series(buckets)
        valid_buckets = list(bucket_series.dropna().unique())
        if not valid_buckets:
            return np.sort(rng.choice(n, size=k, replace=False))

        selected: list[int] = []
        per_bucket = max(1, k // len(valid_buckets))
        for bucket in sorted(valid_buckets):
            bucket_idx = np.flatnonzero(bucket_series.to_numpy() == bucket)
            take = min(per_bucket, len(bucket_idx), k - len(selected))
            if take > 0:
                selected.extend(rng.choice(bucket_idx, size=take, replace=False).tolist())
            if len(selected) >= k:
                break

        if len(selected) < k:
            remaining = np.setdiff1d(np.arange(n), np.array(selected, dtype=int), assume_unique=False)
            selected.extend(rng.choice(remaining, size=k - len(selected), replace=False).tolist())
        return np.sort(np.array(selected, dtype=int))

    raise ValueError(f"Unknown calibration policy: {config.policy}")


def summarize_calibration_policies(
    meta: pd.DataFrame,
    targets: np.ndarray,
    policies: list[str],
    num_windows: int,
    volatility_window: int = 20,
    seed: int = 7,
    tail_quantile: float = 0.9,
    inputs: np.ndarray | None = None,
) -> dict[str, dict[str, float | str]]:
    target = targets.reshape(-1)
    abs_target = np.abs(target)
    tail_threshold = float(np.quantile(abs_target, tail_quantile)) if len(abs_target) else 0.0
    full_start = pd.Timestamp(meta["date"].min())
    full_end = pd.Timestamp(meta["date"].max())
    full_days = max(1, (full_end - full_start).days)

    summaries: dict[str, dict[str, float | str]] = {}
    for policy in policies:
        cfg = CalibrationConfig(
            policy=policy,
            num_windows=num_windows,
            volatility_window=volatility_window,
            seed=seed,
        )
        idx = select_calibration_indices(meta, target, cfg, inputs=inputs)
        selected_abs = abs_target[idx]
        selected_meta = meta.iloc[idx]
        start = pd.Timestamp(selected_meta["date"].min())
        end = pd.Timestamp(selected_meta["date"].max())
        coverage_days = max(0, (end - start).days)
        summaries[policy] = {
            "num_windows": float(len(idx)),
            "date_start": start.date().isoformat(),
            "date_end": end.date().isoformat(),
            "date_coverage_ratio": float(coverage_days / full_days),
            "abs_target_mean": float(np.mean(selected_abs)),
            "abs_target_p50": float(np.quantile(selected_abs, 0.5)),
            "abs_target_p90": float(np.quantile(selected_abs, 0.9)),
            "abs_target_p99": float(np.quantile(selected_abs, 0.99)),
            "tail_coverage_rate": float(np.mean(selected_abs >= tail_threshold)),
            "unique_assets": float(selected_meta["asset"].nunique()),
        }
    return summaries


def input_window_volatility(inputs: np.ndarray | None, window: int) -> np.ndarray:
    if inputs is None:
        raise ValueError("volatility_stratified calibration requires input windows.")
    x = np.asarray(inputs)
    if x.ndim == 3:
        # (n_windows, context_length, num_features): volatility proxy is the
        # mean |return| (feature 0) over the last `window` time steps. A flat
        # slice here would mix the trailing features of the last ~2 steps
        # instead of a `window`-day lookback.
        lookback = min(max(1, window), x.shape[1])
        return np.mean(np.abs(x[:, -lookback:, 0]), axis=1)
    x = x.reshape(len(x), -1)
    lookback = min(max(1, window), x.shape[1])
    return np.mean(np.abs(x[:, -lookback:]), axis=1)


def rolling_abs_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(np.abs(values)).rolling(window=window, min_periods=1).mean().to_numpy()
