import sys
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from torch import nn


def _ensure_tslib() -> None:
    repo = Path(__file__).resolve().parent.parent.parent.parent
    tslib = repo / "Time-Series-Library"
    if tslib.exists() and str(tslib) not in sys.path:
        sys.path.insert(0, str(tslib))


@dataclass(frozen=True)
class DLinearConfig:
    context_length: int
    horizon: int
    num_channels: int = 1
    moving_avg: int = 25
    individual: bool = False


class TSLibDLinearAdapter(nn.Module):
    def __init__(self, config: DLinearConfig) -> None:
        super().__init__()
        _ensure_tslib()
        from models.DLinear import Model as TSLibDLinearModel

        self.config = config
        tslib_cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=config.context_length,
            pred_len=config.horizon,
            enc_in=config.num_channels,
            moving_avg=config.moving_avg,
        )
        self.model = TSLibDLinearModel(tslib_cfg, individual=config.individual)

    def forward(self, x: Any) -> Any:
        out = self.model(x, None, None, None)  # (B, pred_len, channels)
        return out[:, :, 0]  # (B, pred_len) — first channel