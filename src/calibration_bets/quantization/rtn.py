from __future__ import annotations

import copy
from collections.abc import Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F


class FakeQuantLinear(nn.Module):
    """Linear layer with fixed activation scale and fake-quantized weights."""

    def __init__(self, linear: nn.Linear, activation_amax: float, num_bits: int = 8) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.num_bits = num_bits
        qmax = float(2 ** (num_bits - 1) - 1)
        self.register_buffer("activation_scale", make_scale(torch.tensor(float(activation_amax)), qmax))
        self.register_buffer("weight_qdq", fake_quantize_weight_per_channel(linear.weight.detach(), qmax))
        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_qdq = fake_quantize_activation(x, self.activation_scale, self.num_bits)
        return F.linear(x_qdq, self.weight_qdq, self.bias)


class RTNWeightOnlyLinear(nn.Module):
    """Linear layer with round-to-nearest quantized weights and float activations."""

    def __init__(self, linear: nn.Linear, num_bits: int = 4) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.num_bits = num_bits
        qmax = float(2 ** (num_bits - 1) - 1)
        self.register_buffer("weight_qdq", fake_quantize_weight_per_channel(linear.weight.detach(), qmax))
        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_qdq, self.bias)


class FakeQuantConv1d(nn.Module):
    """Conv1d with fixed activation scale and fake-quantized weights.

    Wraps (a copy of) the original module rather than re-implementing the
    forward pass, so padding semantics are preserved — TSLib's TokenEmbedding
    uses circular padding, which a bare F.conv1d call would silently turn
    into zero padding.
    """

    def __init__(self, conv: nn.Conv1d, activation_amax: float, num_bits: int = 8) -> None:
        super().__init__()
        self.num_bits = num_bits
        qmax = float(2 ** (num_bits - 1) - 1)
        self.register_buffer("activation_scale", make_scale(torch.tensor(float(activation_amax)), qmax))
        self.conv = copy.deepcopy(conv)
        with torch.no_grad():
            self.conv.weight.copy_(fake_quantize_weight_per_channel(conv.weight.detach(), qmax))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(fake_quantize_activation(x, self.activation_scale, self.num_bits))


class FakeQuantGRU(nn.Module):
    """Single-layer unidirectional GRU with fake-quantized weights and activations.

    Unrolls the recurrence so that both matmul inputs (x_t and h_{t-1}) are
    fake-quantized at every step, matching the W-A semantics applied to Linear
    layers. Without this, RNN cores silently stay in fp32 and the architecture
    comparison measures different quantization coverage.
    """

    def __init__(
        self,
        gru: nn.GRU,
        input_amax: float,
        hidden_amax: float,
        num_bits: int = 8,
    ) -> None:
        super().__init__()
        if gru.num_layers != 1 or gru.bidirectional or not gru.batch_first:
            raise NotImplementedError("FakeQuantGRU supports single-layer unidirectional batch_first GRUs.")
        self.hidden_size = gru.hidden_size
        self.num_bits = num_bits
        qmax = float(2 ** (num_bits - 1) - 1)
        self.register_buffer("x_scale", make_scale(torch.tensor(float(input_amax)), qmax))
        self.register_buffer("h_scale", make_scale(torch.tensor(float(hidden_amax)), qmax))
        self.register_buffer("weight_ih_qdq", fake_quantize_weight_per_channel(gru.weight_ih_l0.detach(), qmax))
        self.register_buffer("weight_hh_qdq", fake_quantize_weight_per_channel(gru.weight_hh_l0.detach(), qmax))
        bias_ih = getattr(gru, "bias_ih_l0", None)
        bias_hh = getattr(gru, "bias_hh_l0", None)
        self.register_buffer("bias_ih", bias_ih.detach().clone() if bias_ih is not None else None)
        self.register_buffer("bias_hh", bias_hh.detach().clone() if bias_hh is not None else None)

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size) if hx is None else hx[0]
        outputs = []
        for t in range(steps):
            xt = fake_quantize_activation(x[:, t, :], self.x_scale, self.num_bits)
            hq = fake_quantize_activation(h, self.h_scale, self.num_bits)
            gi = xt @ self.weight_ih_qdq.t()
            gh = hq @ self.weight_hh_qdq.t()
            if self.bias_ih is not None:
                gi = gi + self.bias_ih
            if self.bias_hh is not None:
                gh = gh + self.bias_hh
            i_r, i_z, i_n = gi.chunk(3, dim=1)
            h_r, h_z, h_n = gh.chunk(3, dim=1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1.0 - z) * n + z * h
            outputs.append(h)
        return torch.stack(outputs, dim=1), h.unsqueeze(0)


def fake_static_int8(
    model: nn.Module,
    calibration_batches: Iterable[torch.Tensor],
    num_bits: int = 8,
    layer_filter: "Callable[[str], bool] | None" = None,
    percentile: float | None = None,
) -> nn.Module:
    """Static fake-quant of all matmuls.

    `layer_filter(name)` restricts which layers get quantized (only-one /
    leave-one-out sensitivity studies). `percentile` (e.g. 0.999) switches the
    activation scale from abs-max to a calibration percentile — trading a
    little clipping for resolution on concentrated distributions.
    """
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    qmodel = copy.deepcopy(model).cpu().eval()
    activation_amax = collect_linear_input_amax(qmodel, calibration_batches, percentile=percentile)
    replace_linear_modules(qmodel, activation_amax, num_bits=num_bits, layer_filter=layer_filter)
    replace_gru_modules(qmodel, activation_amax, num_bits=num_bits, layer_filter=layer_filter)
    replace_conv1d_modules(qmodel, activation_amax, num_bits=num_bits, layer_filter=layer_filter)
    return qmodel.eval()


def fake_static_mixed(
    model: nn.Module,
    calibration_batches: Iterable[torch.Tensor],
    protect: frozenset[str] | set[str],
    low_bits: int = 4,
    high_bits: int = 8,
    percentile: float | None = None,
) -> nn.Module:
    """Mixed precision: layers named in `protect` at high_bits, rest at low_bits."""
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    qmodel = copy.deepcopy(model).cpu().eval()
    activation_amax = collect_linear_input_amax(qmodel, calibration_batches, percentile=percentile)
    protect = set(protect)
    for bits, flt in ((high_bits, lambda n: n in protect), (low_bits, lambda n: n not in protect)):
        replace_linear_modules(qmodel, activation_amax, num_bits=bits, layer_filter=flt)
        replace_gru_modules(qmodel, activation_amax, num_bits=bits, layer_filter=flt)
        replace_conv1d_modules(qmodel, activation_amax, num_bits=bits, layer_filter=flt)
    return qmodel.eval()


def rtn_w4(
    model: nn.Module,
    num_bits: int = 4,
) -> nn.Module:
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    qmodel = copy.deepcopy(model).cpu().eval()
    replace_linear_modules_weight_only(qmodel, num_bits=num_bits)
    quantize_conv1d_weights_inplace(qmodel, num_bits=num_bits)
    quantize_gru_weights_inplace(qmodel, num_bits=num_bits)
    return qmodel.eval()


@torch.no_grad()
def quantize_conv1d_weights_inplace(model: nn.Module, num_bits: int = 4) -> None:
    """Weight-only RTN for Conv1d: qdq weights in place, preserving the module
    (and therefore its padding mode) instead of re-implementing the forward."""
    qmax = float(2 ** (num_bits - 1) - 1)
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            module.weight.copy_(fake_quantize_weight_per_channel(module.weight.detach(), qmax))


@torch.no_grad()
def quantize_gru_weights_inplace(model: nn.Module, num_bits: int = 4) -> None:
    """Weight-only RTN for GRU cores: qdq the recurrent weight tensors in place."""
    qmax = float(2 ** (num_bits - 1) - 1)
    for module in model.modules():
        if isinstance(module, nn.GRU):
            for pname, param in module.named_parameters():
                if pname.startswith("weight_"):
                    param.copy_(fake_quantize_weight_per_channel(param.detach(), qmax))


def _abs_stat(x: torch.Tensor, percentile: float | None) -> float:
    """abs-max of a tensor, or an abs-percentile when `percentile` is set."""
    ax = x.detach().abs().flatten().float()
    if percentile is None:
        return float(ax.max().cpu())
    if ax.numel() > 4_000_000:  # torch.quantile size limit; stride-subsample
        ax = ax[:: ax.numel() // 4_000_000 + 1]
    return float(torch.quantile(ax, float(percentile)).cpu())


@torch.no_grad()
def collect_linear_input_amax(
    model: nn.Module,
    calibration_batches: Iterable[torch.Tensor],
    percentile: float | None = None,
) -> dict[str, float]:
    """Per-layer activation scale statistic over the calibration set.

    Default is global abs-max. With `percentile`, each batch contributes its
    abs-percentile and the max over batches is kept — a conservative
    approximation of the global percentile.
    """
    activation_amax: dict[str, float] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            current = _abs_stat(x, percentile)
            activation_amax[name] = max(activation_amax.get(name, 0.0), current)

        return hook

    def make_gru_hook(name: str):
        # input sequence amax under <name>.x; hidden-state amax under <name>.h,
        # taken from the output sequence (= h_t at every step, so it bounds
        # both h_{t-1} inputs and any initial hx produced by a previous call)
        def hook(_module: nn.Module, inputs: tuple, output: tuple) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                cur = _abs_stat(inputs[0], percentile)
                key = f"{name}.x"
                activation_amax[key] = max(activation_amax.get(key, 0.0), cur)
            out_seq = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(out_seq):
                cur = _abs_stat(out_seq, percentile)
                key = f"{name}.h"
                activation_amax[key] = max(activation_amax.get(key, 0.0), cur)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            handles.append(module.register_forward_pre_hook(make_hook(name)))
        elif isinstance(module, nn.GRU):
            handles.append(module.register_forward_hook(make_gru_hook(name)))

    try:
        for batch in calibration_batches:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(x.cpu())
    finally:
        for handle in handles:
            handle.remove()

    return activation_amax


def replace_linear_modules(
    module: nn.Module,
    activation_amax: dict[str, float],
    prefix: str = "",
    num_bits: int = 8,
    layer_filter: "Callable[[str], bool] | None" = None,
) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, (FakeQuantLinear, FakeQuantConv1d, FakeQuantGRU)):
            continue
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and should_replace_linear(full_name, child):
            if layer_filter is None or layer_filter(full_name):
                amax = activation_amax.get(full_name, 1.0)
                setattr(module, child_name, FakeQuantLinear(child, amax, num_bits=num_bits))
        else:
            replace_linear_modules(
                child, activation_amax, prefix=full_name, num_bits=num_bits, layer_filter=layer_filter
            )


def replace_gru_modules(
    module: nn.Module,
    activation_amax: dict[str, float],
    prefix: str = "",
    num_bits: int = 8,
    layer_filter: "Callable[[str], bool] | None" = None,
) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, (FakeQuantLinear, FakeQuantConv1d, FakeQuantGRU)):
            continue
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.GRU):
            if layer_filter is None or layer_filter(full_name):
                setattr(
                    module,
                    child_name,
                    FakeQuantGRU(
                        child,
                        input_amax=activation_amax.get(f"{full_name}.x", 1.0),
                        hidden_amax=activation_amax.get(f"{full_name}.h", 1.0),
                        num_bits=num_bits,
                    ),
                )
        else:
            replace_gru_modules(
                child, activation_amax, prefix=full_name, num_bits=num_bits, layer_filter=layer_filter
            )


def replace_linear_modules_weight_only(
    module: nn.Module,
    prefix: str = "",
    num_bits: int = 4,
) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, (FakeQuantLinear, FakeQuantConv1d, FakeQuantGRU)):
            continue
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and should_replace_linear(full_name, child):
            setattr(module, child_name, RTNWeightOnlyLinear(child, num_bits=num_bits))
        else:
            replace_linear_modules_weight_only(child, prefix=full_name, num_bits=num_bits)


def replace_conv1d_modules(
    module: nn.Module,
    activation_amax: dict[str, float],
    prefix: str = "",
    num_bits: int = 8,
    layer_filter: "Callable[[str], bool] | None" = None,
) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, (FakeQuantLinear, FakeQuantConv1d, FakeQuantGRU)):
            continue
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Conv1d):
            if layer_filter is None or layer_filter(full_name):
                amax = activation_amax.get(full_name, 1.0)
                setattr(module, child_name, FakeQuantConv1d(child, amax, num_bits=num_bits))
        else:
            replace_conv1d_modules(
                child, activation_amax, prefix=full_name, num_bits=num_bits, layer_filter=layer_filter
            )


def should_replace_linear(name: str, module: nn.Linear) -> bool:
    # torch.nn.MultiheadAttention accesses out_proj.weight and out_proj.bias
    # directly, so replacing that submodule breaks its forward path.
    if name.endswith("self_attn.out_proj"):
        return False
    if type(module).__name__ == "NonDynamicallyQuantizableLinear":
        return False
    return True


def fake_quantize_activation(x: torch.Tensor, scale: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    qmax = float(2 ** (num_bits - 1) - 1)
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def fake_quantize_weight_per_channel(weight: torch.Tensor, qmax: float) -> torch.Tensor:
    reduce_dims = tuple(range(1, weight.ndim))
    amax = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = make_scale(amax, qmax)
    return torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale


def make_scale(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    eps = torch.finfo(torch.float32).eps
    return torch.clamp(amax.float() / qmax, min=eps)
