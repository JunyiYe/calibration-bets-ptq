from __future__ import annotations

from typing import Any

from torch import nn

from calibration_bets.models.dlinear import DLinearConfig, TSLibDLinearAdapter
from calibration_bets.models.patchtst import PatchTSTBackendConfig, TSLibPatchTSTAdapter
from calibration_bets.models.timemixer import TSLibTimeMixerAdapter, TimeMixerConfig
from calibration_bets.models.tslib_transformer import TSLibTransformerAdapter, TransformerConfig
from calibration_bets.models.tsmixer import TSLibTSMixerAdapter, TSMixerConfig
from calibration_bets.models.itransformer import TSLibiTransformerAdapter, iTransformerConfig
from calibration_bets.models.segrnn import TSLibSegRNNAdapter, SegRNNConfig


def build_model(model_cfg: dict[str, Any], task_cfg: dict[str, Any], num_features: int = 1) -> nn.Module:
    name = model_cfg.get("name", "patchtst")
    if name == "patchtst":
        d_model = int(model_cfg.get("d_model", 128))
        return TSLibPatchTSTAdapter(
            PatchTSTBackendConfig(
                context_length=int(task_cfg["context_length"]),
                horizon=int(task_cfg["horizon"]),
                num_channels=num_features,
                d_model=d_model,
                n_heads=int(model_cfg.get("n_heads", 16)),
                n_layers=int(model_cfg.get("n_layers", 3)),
                d_ff=int(model_cfg.get("d_ff", 4 * d_model)),
                dropout=float(model_cfg.get("dropout", 0.2)),
                patch_length=int(model_cfg.get("patch_length", 16)),
                stride=int(model_cfg.get("stride", 8)),
                activation=str(model_cfg.get("activation", "gelu")),
                factor=int(model_cfg.get("factor", 1)),
            )
        )
    if name == "dlinear":
        return TSLibDLinearAdapter(
            DLinearConfig(
                context_length=int(task_cfg["context_length"]),
                horizon=int(task_cfg["horizon"]),
                num_channels=num_features,
                moving_avg=int(model_cfg.get("moving_avg", 25)),
                individual=bool(model_cfg.get("individual", False)),
            )
        )
    if name == "transformer":
        d_model = int(model_cfg.get("d_model", 32))
        return TSLibTransformerAdapter(
            TransformerConfig(
                context_length=int(task_cfg["context_length"]),
                horizon=int(task_cfg["horizon"]),
                num_channels=num_features,
                d_model=d_model,
                n_heads=int(model_cfg.get("n_heads", 4)),
                n_layers=int(model_cfg.get("n_layers", 2)),
                d_ff=int(model_cfg.get("d_ff", 4 * d_model)),
                dropout=float(model_cfg.get("dropout", 0.3)),
            )
        )
    if name == "timemixer":
        return TSLibTimeMixerAdapter(
            TimeMixerConfig(
                context_length=int(task_cfg["context_length"]),
                horizon=int(task_cfg["horizon"]),
                num_channels=num_features,
                d_model=int(model_cfg.get("d_model", 32)),
                n_heads=int(model_cfg.get("n_heads", 4)),
                n_layers=int(model_cfg.get("n_layers", 2)),
                d_ff=int(model_cfg.get("d_ff", 128)),
                dropout=float(model_cfg.get("dropout", 0.3)),
                moving_avg=int(model_cfg.get("moving_avg", 25)),
                down_sampling_method=str(model_cfg.get("down_sampling_method", "avg")),
                down_sampling_window=int(model_cfg.get("down_sampling_window", 2)),
                down_sampling_layers=int(model_cfg.get("down_sampling_layers", 2)),
                use_norm=int(model_cfg.get("use_norm", 1)),
                channel_independence=bool(model_cfg.get("channel_independence", False)),
            )
        )
    if name == "segrnn":
        return TSLibSegRNNAdapter(
            SegRNNConfig(
                context_length=int(task_cfg["context_length"]),
                horizon=int(task_cfg["horizon"]),
                num_channels=num_features,
                d_model=int(model_cfg.get("d_model", 64)),
                n_layers=int(model_cfg.get("n_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.3)),
            )
        )
    if name == "itransformer":
        return TSLibiTransformerAdapter(
            iTransformerConfig(
                context_length=int(task_cfg["context_length"]),
                horizon=int(task_cfg["horizon"]),
                num_channels=num_features,
                d_model=int(model_cfg.get("d_model", 32)),
                n_heads=int(model_cfg.get("n_heads", 4)),
                n_layers=int(model_cfg.get("n_layers", 2)),
                d_ff=int(model_cfg.get("d_ff", 128)),
                dropout=float(model_cfg.get("dropout", 0.3)),
            )
        )
    if name == "tsmixer":
        return TSLibTSMixerAdapter(
            TSMixerConfig(
                context_length=int(task_cfg["context_length"]),
                horizon=int(task_cfg["horizon"]),
                num_channels=num_features,
                d_model=int(model_cfg.get("d_model", 64)),
                n_layers=int(model_cfg.get("n_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.3)),
            )
        )
    raise ValueError(f"Unsupported model for initial scaffold: {name}")
