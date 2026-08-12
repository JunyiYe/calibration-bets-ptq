from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PanelSpec:
    path: str
    date_col: str = "date"
    asset_col: str = "symbol"
    value_col: str = "return"
    min_history: int = 0


def load_csv_panel(spec: PanelSpec) -> pd.DataFrame:
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Panel CSV not found: {path}. Update the config data.path or create this file."
        )

    df = pd.read_csv(path)
    required = {spec.date_col, spec.asset_col, spec.value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df[[spec.date_col, spec.asset_col, spec.value_col]].copy()
    df[spec.date_col] = pd.to_datetime(df[spec.date_col])
    df = df.sort_values([spec.asset_col, spec.date_col])
    if spec.min_history > 0:
        counts = df.groupby(spec.asset_col)[spec.value_col].transform("size")
        df = df.loc[counts >= spec.min_history]
    return df.rename(
        columns={
            spec.date_col: "date",
            spec.asset_col: "asset",
            spec.value_col: "value",
        }
    )


def chronological_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: str | None = None,
    train_start: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date. train_start is EXCLUSIVE (like the val/test lower
    bounds), so a fixed N-year training window ending Y-2 uses
    train_start = f"{Y-2-N}-12-31". None keeps the expanding-window
    behavior (train = everything through train_end)."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end) if test_end else df["date"].max()

    train = df[df["date"] <= train_end_ts]
    if train_start is not None:
        train = train[train["date"] > pd.Timestamp(train_start)]
    val = df[(df["date"] > train_end_ts) & (df["date"] <= val_end_ts)]
    test = df[(df["date"] > val_end_ts) & (df["date"] <= test_end_ts)]
    return train, val, test
