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
class SegRNNConfig:
    context_length: int
    horizon: int
    num_channels: int = 1
    d_model: int = 64
    n_layers: int = 2
    dropout: float = 0.3


class TSLibSegRNNAdapter(nn.Module):
    def __init__(self, config: SegRNNConfig) -> None:
        super().__init__()
        _ensure_tslib()
        from models.SegRNN import Model as TSLibSegRNNModel

        self.config = config
        tslib_cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=config.context_length,
            pred_len=config.horizon,
            seg_len=1,
            enc_in=config.num_channels,
            d_model=config.d_model,
            e_layers=config.n_layers,
            dropout=config.dropout,
        )
        self.model = TSLibSegRNNModel(tslib_cfg)

    def forward(self, x: Any) -> Any:
        out = self.model(x, None, None, None)  # (B, pred_len, channels)
        return out[:, :, 0]  # (B, pred_len) — first channel
