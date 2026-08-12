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
class TimeMixerConfig:
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
    moving_avg: int = 25
    down_sampling_method: str = "avg"
    down_sampling_window: int = 2
    down_sampling_layers: int = 2
    use_norm: int = 1
    channel_independence: bool = False


class TSLibTimeMixerAdapter(nn.Module):
    def __init__(self, config: TimeMixerConfig) -> None:
        super().__init__()
        _ensure_tslib()
        from models.TimeMixer import Model as TSLibTimeMixerModel

        self.config = config
        tslib_cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=config.context_length,
            label_len=config.context_length // 2,
            pred_len=config.horizon,
            enc_in=config.num_channels,
            c_out=1,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            factor=config.factor,
            embed="fixed",
            freq="h",
            moving_avg=config.moving_avg,
            decomp_method="moving_avg",
            top_k=5,
            num_class=0,
            down_sampling_method=config.down_sampling_method,
            down_sampling_window=config.down_sampling_window,
            down_sampling_layers=config.down_sampling_layers,
            use_norm=config.use_norm,
            channel_independence=config.channel_independence,
        )
        self.model = TSLibTimeMixerModel(tslib_cfg)

    def forward(self, x: Any) -> Any:
        out = self.model(x, None, None, None)  # (B, pred_len, channels)
        return out[:, :, 0]  # (B, pred_len) — first channel