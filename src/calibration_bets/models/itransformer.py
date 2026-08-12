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
class iTransformerConfig:
    context_length: int
    horizon: int
    num_channels: int = 1
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.3
    activation: str = "gelu"
    factor: int = 1


class TSLibiTransformerAdapter(nn.Module):
    def __init__(self, config: iTransformerConfig) -> None:
        super().__init__()
        _ensure_tslib()
        from models.iTransformer import Model as TSLibiTransformerModel

        self.config = config
        tslib_cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=config.context_length,
            pred_len=config.horizon,
            enc_in=config.num_channels,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            factor=config.factor,
            embed="fixed",
            freq="h",
            num_class=0,
        )
        self.model = TSLibiTransformerModel(tslib_cfg)

    def forward(self, x: Any) -> Any:
        out = self.model(x, None, None, None)  # (B, pred_len, 1)
        return out[:, :, 0]  # (B, pred_len) — first channel