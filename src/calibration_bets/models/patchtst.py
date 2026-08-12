from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from torch import nn


@dataclass(frozen=True)
class PatchTSTBackendConfig:
    context_length: int
    horizon: int
    num_channels: int = 1
    d_model: int = 128
    n_heads: int = 16
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.2
    patch_length: int = 16
    stride: int = 8
    activation: str = "gelu"
    factor: int = 1


class TSLibPatchTSTAdapter(nn.Module):
    """Thin adapter around the upstream Time-Series-Library PatchTST model."""

    def __init__(self, config: PatchTSTBackendConfig) -> None:
        super().__init__()
        self._ensure_tslib()
        from models.PatchTST import Model as TSLibPatchTSTModel

        self.config = config
        tslib_cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=config.context_length,
            pred_len=config.horizon,
            enc_in=config.num_channels,
            c_out=1,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
        )
        self.model = TSLibPatchTSTModel(tslib_cfg, patch_len=config.patch_length, stride=config.stride)

    @staticmethod
    def _ensure_tslib() -> None:
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent.parent.parent
        tslib = repo / "Time-Series-Library"
        if tslib.exists() and str(tslib) not in sys.path:
            sys.path.insert(0, str(tslib))

    def forward(self, x: Any) -> Any:
        out = self.model(x, None, None, None)  # (B, pred_len, num_features)
        return out[:, :, 0]  # (B, pred_len) — only the return channel
