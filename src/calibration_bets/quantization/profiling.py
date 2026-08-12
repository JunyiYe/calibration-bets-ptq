"""Per-layer activation statistics and clip-rate probes for the mechanism study.

Two passes per model:
  1. fp32: distribution stats (kurtosis, amax, tail share) per quantizable
     matmul input, plus per-window amax — tests whether fragility tracks
     input-pathway outliers / tokenization granularity.
  2. quantized: per-window fraction of activation elements clipped at each
     FakeQuant layer — aggregated by date, this links quantization error to
     market regimes.

Models may fold channels into the leading dim (SegRNN, PatchTST reshape
(B, C, ...) -> (B*C, ...)); per-window reduction infers the fold factor from
the known batch size, which works because torch reshapes are row-major.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from calibration_bets.quantization.rtn import FakeQuantConv1d, FakeQuantGRU, FakeQuantLinear, should_replace_linear


class _LayerMoments:
    """Running raw moments (float64) for one layer's activation elements."""

    def __init__(self) -> None:
        self.n = 0.0
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.amax = 0.0

    def update(self, x: torch.Tensor) -> None:
        xd = x.detach().double()
        self.n += float(xd.numel())
        self.s1 += float(xd.sum())
        self.s2 += float((xd**2).sum())
        self.s3 += float((xd**3).sum())
        self.s4 += float((xd**4).sum())
        self.amax = max(self.amax, float(xd.abs().max()))

    def summary(self) -> dict[str, float]:
        if self.n == 0:
            return {}
        m1 = self.s1 / self.n
        m2 = self.s2 / self.n
        m3 = self.s3 / self.n
        m4 = self.s4 / self.n
        var = max(m2 - m1 * m1, 1e-24)
        kurt = (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1**4) / (var * var) - 3.0
        return {
            "n_elements": self.n,
            "mean": m1,
            "std": var**0.5,
            "excess_kurtosis": kurt,
            "amax": self.amax,
            "amax_over_std": self.amax / var**0.5,
        }


def _per_window_reduce(x: torch.Tensor, batch_size: int, op: str, threshold: float = 0.0) -> np.ndarray | None:
    """Reduce a (B*k, ...) activation to one value per window (B,)."""
    lead = x.shape[0]
    if batch_size <= 0 or lead % batch_size != 0:
        return None
    flat = x.detach().reshape(batch_size, -1)
    if op == "amax":
        return flat.abs().amax(dim=1).float().cpu().numpy()
    if op == "clip_frac":
        return (flat.abs() > threshold).double().mean(dim=1).cpu().numpy()
    raise ValueError(op)


def quantizable_input_points(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """(name, module) for every matmul whose input the static path quantizes."""
    points = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_replace_linear(name, module):
            points.append((name, module))
        elif isinstance(module, (nn.Conv1d, nn.GRU)):
            points.append((name, module))
    return points


@torch.no_grad()
def profile_fp32(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], np.ndarray]:
    """Returns (per-layer stats, per-layer per-window amax, predictions)."""
    model = model.to(device).eval()
    moments: dict[str, _LayerMoments] = {}
    window_amax: dict[str, list[np.ndarray]] = {}
    pending: dict[str, list[np.ndarray]] = {}
    holder = {"B": 0}
    handles = []

    def make_hook(name: str):
        def hook(_m: nn.Module, inputs: tuple) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            x = inputs[0]
            moments.setdefault(name, _LayerMoments()).update(x)
            reduced = _per_window_reduce(x, holder["B"], "amax")
            if reduced is not None:
                pending.setdefault(name, []).append(reduced)

        return hook

    for name, module in quantizable_input_points(model):
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    preds = []
    try:
        for x, _y in loader:
            holder["B"] = x.shape[0]
            preds.append(model(x.to(device)).detach().cpu().numpy())
            # a layer may run several times per forward (e.g. SegRNN calls its
            # GRU twice); combine within-batch calls so arrays stay aligned
            # one-row-per-window
            for name, calls in pending.items():
                window_amax.setdefault(name, []).append(np.max(calls, axis=0))
            pending.clear()
    finally:
        for h in handles:
            h.remove()

    stats = {k: v.summary() for k, v in moments.items()}
    amax_arrays = {
        k: np.concatenate(v) for k, v in window_amax.items() if sum(len(a) for a in v)
    }
    return stats, amax_arrays, np.concatenate(preds, axis=0)


@torch.no_grad()
def profile_clipping(
    qmodel: nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-window clip fraction at every FakeQuant layer of a quantized model."""
    qmodel = qmodel.to(device).eval()
    clip: dict[str, list[np.ndarray]] = {}
    pending: dict[str, list[np.ndarray]] = {}
    holder = {"B": 0}
    handles = []

    def make_hook(name: str, threshold: float):
        def hook(_m: nn.Module, inputs: tuple) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            reduced = _per_window_reduce(inputs[0], holder["B"], "clip_frac", threshold)
            if reduced is not None:
                pending.setdefault(name, []).append(reduced)

        return hook

    for name, module in qmodel.named_modules():
        if isinstance(module, (FakeQuantLinear, FakeQuantConv1d)):
            qmax = float(2 ** (module.num_bits - 1) - 1)
            threshold = float(module.activation_scale) * qmax
            handles.append(module.register_forward_pre_hook(make_hook(name, threshold)))
        elif isinstance(module, FakeQuantGRU):
            qmax = float(2 ** (module.num_bits - 1) - 1)
            threshold = float(module.x_scale) * qmax
            handles.append(module.register_forward_pre_hook(make_hook(name, threshold)))

    preds = []
    try:
        for x, _y in loader:
            holder["B"] = x.shape[0]
            preds.append(qmodel(x.to(device)).detach().cpu().numpy())
            for name, calls in pending.items():
                clip.setdefault(name, []).append(np.mean(calls, axis=0))
            pending.clear()
    finally:
        for h in handles:
            h.remove()

    clip_arrays = {k: np.concatenate(v) for k, v in clip.items()}
    return clip_arrays, np.concatenate(preds, axis=0)
