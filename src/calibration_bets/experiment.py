"""Shared experiment assembly: config -> window datasets and calibration loaders.

Every script (training, significance tests, noise/quantization analyses) must
see identical windows; this module is the single place that turns a config
into datasets so the pipelines cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader, Subset

from calibration_bets.data.features import add_return_features, compute_panel_stats, normalize_panel
from calibration_bets.data.panel import PanelSpec, chronological_split, load_csv_panel
from calibration_bets.data.windowing import TimeSeriesWindowDataset, WindowConfig
from calibration_bets.quantization.calibration import CalibrationConfig, select_calibration_indices


def parse_range(v: Any) -> tuple[float, float] | None:
    if v is None:
        return None
    parts = str(v).split(",")
    return (float(parts[0]), float(parts[1]))


@dataclass
class ExperimentData:
    feature_columns: tuple[str, ...]
    window_config: WindowConfig
    splits: dict[str, TimeSeriesWindowDataset]


def build_datasets(
    cfg: dict[str, Any],
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> ExperimentData:
    """Assemble window datasets from an experiment config.

    Pipeline order: derived features on the raw panel (backward-looking only),
    chronological split, train-stats z-score normalization, then windowing.
    Split frames are always computed from the full panel so normalization
    stats are identical regardless of which `splits` are requested.
    """
    data_cfg = cfg["data"]
    if data_cfg.get("kind") != "csv_panel":
        raise ValueError("Only data.kind=csv_panel is supported.")
    task_cfg = cfg["task"]

    panel = load_csv_panel(
        PanelSpec(
            path=data_cfg["path"],
            date_col=data_cfg.get("date_col", "date"),
            asset_col=data_cfg.get("asset_col", "symbol"),
            value_col=data_cfg.get("value_col", "return"),
            min_history=int(data_cfg.get("min_history", 0)),
        )
    )

    feat_cols = tuple(task_cfg.get("feature_columns", ["value"]))
    if feat_cols != ("value",):
        panel = add_return_features(panel, feature_columns=feat_cols)

    unknown = set(splits) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"Unknown splits requested: {sorted(unknown)}")

    train_end = pd.Timestamp(data_cfg["train_end"])
    val_end = pd.Timestamp(data_cfg["val_end"])
    test_end = pd.Timestamp(data_cfg["test_end"]) if data_cfg.get("test_end") else panel["date"].max()
    # exclusive lower bound of the training window (fixed-window folds);
    # None = expanding window from the panel start
    train_start = pd.Timestamp(data_cfg["train_start"]) if data_cfg.get("train_start") else None

    # normalization stats always come from the training period, regardless of
    # which splits are requested
    panel_norm = task_cfg.get("normalize", "zscore")
    if len(feat_cols) > 1 and panel_norm != "none":
        train_df, _, _ = chronological_split(
            panel, train_end=train_end, val_end=val_end, test_end=test_end, train_start=train_start
        )
        stats = compute_panel_stats(train_df, list(feat_cols))
        panel = normalize_panel(panel, stats, list(feat_cols))

    # with derived features: normalization is done at panel level (if enabled);
    # without: original per-window normalization behavior
    use_per_window = len(feat_cols) == 1 and panel_norm != "none"
    window_cfg = WindowConfig(
        context_length=int(task_cfg["context_length"]),
        horizon=1,
        forward_period=int(task_cfg.get("forward_period", 1)),
        normalize=panel_norm if use_per_window else "none",
        target_transform=task_cfg.get("target_transform", "raw"),
        target_clip=parse_range(task_cfg.get("target_clip")),
        feature_columns=feat_cols,
        target_agg=task_cfg.get("target_agg", "sum"),
        target_log=bool(task_cfg.get("target_log", False)),
    )

    # Windows are built on the full panel and assigned to splits by TARGET
    # date, so a val/test window's context may span backward across the split
    # boundary (backward-looking data only — no leakage). Building within each
    # split frame instead would silently discard the first
    # context_length+forward_period days of every split (~27% of a 1-year
    # test fold, including e.g. the start of the 2022 bear market).
    if "train" not in splits:
        # efficiency: don't build training-period windows nobody asked for;
        # keep a generous calendar buffer for context that spans the boundary
        earliest_target = train_end if "val" in splits else val_end
        lookback_days = 3 * (window_cfg.context_length + window_cfg.forward_period) + 60
        panel = panel[panel["date"] >= earliest_target - pd.Timedelta(days=lookback_days)]

    full_ds = TimeSeriesWindowDataset(panel[panel["date"] <= test_end], window_cfg)
    bounds = {
        "train": (train_start, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }
    datasets = {name: full_ds.subset_by_date(*bounds[name]) for name in splits}
    return ExperimentData(feature_columns=feat_cols, window_config=window_cfg, splits=datasets)


def build_calibration_loader(
    val_ds: TimeSeriesWindowDataset,
    policy: str,
    calibration_cfg: dict[str, Any],
    seed: int,
    batch_size: int,
) -> DataLoader:
    indices = select_calibration_indices(
        val_ds.meta,
        val_ds.raw_y,
        CalibrationConfig(
            policy=policy,
            num_windows=int(calibration_cfg.get("num_windows", 1024)),
            volatility_window=int(calibration_cfg.get("volatility_window", 20)),
            seed=seed,
        ),
        inputs=val_ds.x,
    )
    return DataLoader(Subset(val_ds, indices.tolist()), batch_size=batch_size, shuffle=False, drop_last=False)
