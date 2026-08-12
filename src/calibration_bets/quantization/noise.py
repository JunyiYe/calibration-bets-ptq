"""Weight-noise baseline for the redundancy control experiment.

If a model's quantization insensitivity were purely parameter redundancy, it
would tolerate ANY perturbation of the same magnitude. This module injects iid
Gaussian noise whose per-channel std matches the second moment of RTN uniform
quantization error (scale^2/12), on exactly the module set that `rtn_w4`
quantizes. Comparing the model's IC under matched noise vs under actual
quantization separates generic perturbation robustness (redundancy) from
structured quantization error.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from calibration_bets.quantization.rtn import make_scale, should_replace_linear


def rtn_noise_sigma(weight: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Per-channel std of uniform quantization error for per-channel RTN."""
    qmax = float(2 ** (num_bits - 1) - 1)
    reduce_dims = tuple(range(1, weight.ndim))
    amax = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    return make_scale(amax, qmax) / math.sqrt(12.0)


@torch.no_grad()
def perturb_weights_like_rtn(
    model: nn.Module,
    num_bits: int = 4,
    multiplier: float = 1.0,
    seed: int = 0,
) -> nn.Module:
    """Return a copy of `model` with quantization-magnitude Gaussian weight noise.

    Perturbs the same modules that `rtn_w4` quantizes (Linear except excluded
    attention projections, plus Conv1d) so the comparison is like-for-like.
    `multiplier` scales the noise std relative to the matched level, giving a
    continuous perturbation-robustness curve through the quantization point.
    """
    qmodel = copy.deepcopy(model).cpu().eval()
    gen = torch.Generator().manual_seed(int(seed))
    for name, module in qmodel.named_modules():
        if isinstance(module, nn.GRU):
            for pname, param in module.named_parameters():
                if pname.startswith("weight_"):
                    sigma = rtn_noise_sigma(param, num_bits) * float(multiplier)
                    param.add_((torch.randn(param.shape, generator=gen) * sigma).to(param.dtype))
            continue
        is_target = (
            isinstance(module, nn.Linear) and should_replace_linear(name, module)
        ) or isinstance(module, nn.Conv1d)
        if not is_target:
            continue
        sigma = rtn_noise_sigma(module.weight, num_bits) * float(multiplier)
        noise = torch.randn(module.weight.shape, generator=gen) * sigma
        module.weight.add_(noise.to(module.weight.dtype))
    return qmodel


def count_perturbed_parameters(model: nn.Module) -> dict[str, int | float]:
    """Report how many modules/weights the quantization/noise path touches."""
    n_modules = 0
    n_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.GRU):
            n_modules += 1
            n_weights += sum(p.numel() for pn, p in module.named_parameters() if pn.startswith("weight_"))
            continue
        is_target = (
            isinstance(module, nn.Linear) and should_replace_linear(name, module)
        ) or isinstance(module, nn.Conv1d)
        if is_target:
            n_modules += 1
            n_weights += module.weight.numel()
    # trainable only: fixed sinusoid embeddings (requires_grad=False) are not
    # part of the learned model and would deflate the coverage ratio
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "perturbed_modules": n_modules,
        "perturbed_weights": n_weights,
        "total_parameters": total,
        "weight_coverage": n_weights / total if total else 0.0,
    }
