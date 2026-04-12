from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.quantization.sharq_ops import (
    global_nvfp4_scale,
    load_sharq_ops,
)


@dataclass
class SharQPreparedActivation:
    backend: Literal["kernel", "reference"]
    scale: torch.Tensor
    a_comp: torch.Tensor | None = None
    e: torch.Tensor | None = None
    sfa_sparse: torch.Tensor | None = None
    q_res: torch.Tensor | None = None
    sf_res: torch.Tensor | None = None
    x_sparse_q: torch.Tensor | None = None
    x_res_q: torch.Tensor | None = None


def apply_rmsnorm(
    x: torch.Tensor, rmsnorm_weight: torch.Tensor, rmsnorm_eps: float
) -> torch.Tensor:
    x_float = x.float()
    weight_float = rmsnorm_weight.float()
    inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + rmsnorm_eps)
    return (x_float * inv_rms * weight_float).to(torch.bfloat16)


def top2_pairs_8_maxabs(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2)
    pair_scores = groups.abs().amax(dim=-1)
    top_idx = pair_scores.topk(k=2, dim=-1).indices
    pair_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
    pair_mask.scatter_(-1, top_idx, True)
    value_mask = pair_mask.unsqueeze(-1).expand_as(groups)
    return (groups * value_mask).reshape_as(x)


def quantize_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    representable_vals = torch.tensor(
        [
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    best = torch.full_like(tensor, representable_vals[0])
    best_diff = torch.abs(tensor - representable_vals[0])
    for value in representable_vals[1:]:
        diff = torch.abs(tensor - value)
        mask = diff < best_diff
        best = torch.where(mask, value, best)
        best_diff = torch.where(mask, diff, best_diff)
    return best


def quantize_ue4m3(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.clamp(tensor, min=2e-3, max=448.0)
    exponent = torch.floor(torch.log2(tensor + 1e-9))
    mantissa_val = tensor / (2**exponent) - 1.0
    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    return (1 + quantized_mantissa_val) * (2**exponent)


def quantize_nvfp4_tensor(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    original_shape = tensor.shape
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))

    reshaped = tensor.view(-1, group_size)
    max_abs = reshaped.abs().max(dim=1, keepdim=True)[0]
    scale = max_abs / 6.0
    scale[scale == 0] = 1e-9
    dq_scale = quantize_ue4m3(scale)
    normalized = reshaped / dq_scale
    q = quantize_e2m1(normalized)
    out = (q * dq_scale).view(tensor.shape)

    if padding != 0:
        out = out[..., :-padding]
    return out.view(original_shape)


def prepare_sharq_reference_activation(x: torch.Tensor) -> SharQPreparedActivation:
    x_bf16 = x.to(torch.bfloat16)
    scale = global_nvfp4_scale(x_bf16)
    x_scaled = x_bf16.float() / scale
    x_sparse = top2_pairs_8_maxabs(x_scaled)
    x_sparse_q = quantize_nvfp4_tensor(x_sparse, group_size=32).to(torch.bfloat16)
    x_res_q = quantize_nvfp4_tensor(x_scaled - x_sparse_q.float(), group_size=16).to(
        torch.bfloat16
    )
    return SharQPreparedActivation(
        backend="reference",
        scale=scale,
        x_sparse_q=x_sparse_q,
        x_res_q=x_res_q,
    )


def prepare_sharq_activation(
    x: torch.Tensor,
    out_features: int,
    *,
    backend: Literal["auto", "kernel", "reference"] = "auto",
) -> SharQPreparedActivation:
    if backend == "reference":
        return prepare_sharq_reference_activation(x)

    if backend == "auto":
        backend = "kernel"

    if backend != "kernel":
        raise ValueError(f"Unsupported SharQ prepare backend: {backend}")

    sharq_ops = load_sharq_ops()
    x_bf16 = x.to(torch.bfloat16).contiguous()
    scale = global_nvfp4_scale(x_bf16)
    x_scaled = (x_bf16 / scale).contiguous()
    a_comp, e, sfa_sparse, q_res, sf_res = sharq_ops.fused_sparse_residual_quantize_x(
        x_scaled, int(out_features)
    )
    return SharQPreparedActivation(
        backend="kernel",
        scale=scale,
        a_comp=a_comp,
        e=e,
        sfa_sparse=sfa_sparse,
        q_res=q_res,
        sf_res=sf_res,
    )


def prepare_sharq_activation_after_rmsnorm(
    x: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    rmsnorm_eps: float,
    out_features: int,
    *,
    backend: Literal["auto", "kernel", "reference"] = "auto",
) -> SharQPreparedActivation:
    if backend == "reference":
        x_norm = apply_rmsnorm(x.to(torch.bfloat16), rmsnorm_weight, rmsnorm_eps)
        return prepare_sharq_reference_activation(x_norm)

    if backend == "auto":
        backend = "kernel"

    if backend != "kernel":
        raise ValueError(f"Unsupported SharQ prepare backend: {backend}")

    sharq_ops = load_sharq_ops()
    a_comp, e, sfa_sparse, q_res, sf_res, scale = (
        sharq_ops.fused_rmsnorm_sparse_residual_quantize_x(
            x.to(torch.bfloat16).contiguous(),
            rmsnorm_weight.to(torch.bfloat16).contiguous(),
            float(rmsnorm_eps),
            int(out_features),
        )
    )
    return SharQPreparedActivation(
        backend="kernel",
        scale=scale,
        a_comp=a_comp,
        e=e,
        sfa_sparse=sfa_sparse,
        q_res=q_res,
        sf_res=sf_res,
    )


__all__ = [
    "SharQPreparedActivation",
    "apply_rmsnorm",
    "prepare_sharq_activation",
    "prepare_sharq_activation_after_rmsnorm",
    "prepare_sharq_reference_activation",
    "quantize_e2m1",
    "quantize_nvfp4_tensor",
    "quantize_ue4m3",
    "top2_pairs_8_maxabs",
]
