"""SmoothQuant-style outlier migration for SegRNN's GRU input path.

Per-channel scales s_j fold into the embedding Linear's output rows and the
GRU's W_ih columns: ReLU(W_e x)/s = ReLU((W_e/s) x) for s > 0, and
(x/s) @ (W_ih * s)^T == x @ W_ih^T, so fp32 outputs are unchanged while the
GRU input's per-channel ranges are equalized for per-tensor activation
quantization.

Scope note (a finding, not a shortcut): the hidden-state path cannot be
smoothed — h_t is produced by the gated nonlinear recurrence, so rescaling it
changes the dynamics. Outlier migration can therefore only ever address the
x-path of a recurrent core.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


@torch.no_grad()
def smooth_segrnn_gru(
    model: nn.Module,
    calibration_batches: list[torch.Tensor],
    alpha: float = 0.5,
) -> nn.Module:
    """Return a copy of a SegRNN adapter with the GRU x-path smoothed."""
    smoothed = copy.deepcopy(model).cpu().eval()
    core = smoothed.model  # TSLib SegRNN
    embed: nn.Linear = core.valueEmbedding[0]
    gru: nn.GRU = core.rnn

    # per-channel |x| max of the GRU input over the calibration set
    act_amax: torch.Tensor | None = None

    def hook(_m: nn.Module, inputs: tuple) -> None:
        nonlocal act_amax
        x = inputs[0]
        cur = x.detach().reshape(-1, x.shape[-1]).abs().amax(dim=0)
        act_amax = cur if act_amax is None else torch.maximum(act_amax, cur)

    handle = gru.register_forward_pre_hook(hook)
    try:
        for batch in calibration_batches:
            smoothed(batch.cpu())
    finally:
        handle.remove()
    if act_amax is None:
        raise RuntimeError("Calibration produced no GRU inputs.")

    w_col_amax = gru.weight_ih_l0.abs().amax(dim=0)
    eps = 1e-8
    s = act_amax.clamp(min=eps).pow(alpha) / w_col_amax.clamp(min=eps).pow(1.0 - alpha)
    # keep scales in a moderate band: extreme per-channel scales are exact in
    # real arithmetic but ill-conditioned in fp32 (weights blown up by 1/s),
    # which was observed to break fp32 equivalence by up to 0.36 on some
    # checkpoints with near-dead channels
    s = s / s.median()
    s = s.clamp(min=0.1, max=10.0)

    embed.weight.div_(s.unsqueeze(1))
    if embed.bias is not None:
        embed.bias.div_(s)
    gru.weight_ih_l0.mul_(s.unsqueeze(0))
    # SegRNN feeds a SECOND input into the same GRU: the decode step consumes
    # cat([pos_emb, channel_emb], dim=-1) directly, so those parameters must
    # carry the inverse scale too or fp32 equivalence breaks
    d = gru.input_size
    half = d // 2
    core.pos_emb.div_(s[:half])
    core.channel_emb.div_(s[half:])
    return smoothed
