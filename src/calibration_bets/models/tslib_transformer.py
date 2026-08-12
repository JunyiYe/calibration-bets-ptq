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
class TransformerConfig:
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


class TSLibTransformerAdapter(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        _ensure_tslib()
        from models.Transformer import Model as TSLibTransformerModel

        self.config = config
        tslib_cfg = SimpleNamespace(
            task_name="anomaly_detection",
            seq_len=config.context_length,
            pred_len=config.horizon,
            enc_in=config.num_channels,
            c_out=1,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.n_layers,
            d_layers=1,
            d_ff=config.d_ff,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
            embed="fixed",
            freq="h",
        )
        self.model = TSLibTransformerModel(tslib_cfg)

    def forward(self, x: Any) -> Any:
        out = self.model(x, None, None, None)  # (B, L, 1)
        return out[:, -1, :]  # (B, 1) — last timestep, keep channel dim
