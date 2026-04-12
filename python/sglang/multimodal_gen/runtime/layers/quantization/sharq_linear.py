from __future__ import annotations

from typing import List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.layers.quantization.sharq_ops import (
    dense_scale_buffer_numel,
    load_sharq_ops,
    sparse_scale_buffer_numel,
)
from sglang.multimodal_gen.runtime.layers.quantization.sharq_prepare import (
    prepare_sharq_activation,
)
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs


def _copy_tensor_param(param: Parameter, loaded_weight: torch.Tensor) -> None:
    if loaded_weight.ndim == 0:
        loaded_weight = loaded_weight.reshape(1)
    assert param.shape == loaded_weight.shape, (
        f"Tried to load SharQ tensor of size {tuple(loaded_weight.shape)} "
        f"into parameter with size {tuple(param.shape)}"
    )
    param.data.copy_(loaded_weight)


class SharQLinearMethod(LinearMethodBase):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype

        if input_size_per_partition % 32 != 0:
            raise ValueError(
                "SharQ requires input features to be a multiple of 32, got "
                f"{input_size_per_partition} for {layer.__class__.__name__}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        sparse_scale_numel = sparse_scale_buffer_numel(
            output_size_per_partition, input_size_per_partition
        )
        dense_scale_numel = dense_scale_buffer_numel(
            output_size_per_partition, input_size_per_partition
        )

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        sfw_sparse = Parameter(
            torch.empty(sparse_scale_numel, dtype=torch.uint8), requires_grad=False
        )
        sfw_dense = Parameter(
            torch.empty(dense_scale_numel, dtype=torch.uint8), requires_grad=False
        )
        weight_scale = Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("sfw_sparse", sfw_sparse)
        layer.register_parameter("sfw_dense", sfw_dense)
        layer.register_parameter("weight_scale", weight_scale)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # SharQ milestone one only supports tp_size == 1, so quantized tensors
        # should be copied as-is instead of going through the generic TP sharder.
        weight_loader = _copy_tensor_param
        set_weight_attrs(
            qweight,
            {
                "weight_loader": weight_loader,
                "missing_param_init": "zeros",
            },
        )
        set_weight_attrs(
            sfw_sparse,
            {
                "weight_loader": weight_loader,
                "missing_param_init": "zeros",
            },
        )
        set_weight_attrs(
            sfw_dense,
            {
                "weight_loader": weight_loader,
                "missing_param_init": "zeros",
            },
        )
        set_weight_attrs(
            weight_scale,
            {
                "weight_loader": weight_loader,
                "missing_param_init": "ones",
            },
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, input_shape[-1]).to(torch.bfloat16).contiguous()

        prepared = prepare_sharq_activation(
            x_2d,
            out_features=layer.output_size_per_partition,
            backend="kernel",
        )
        if prepared.backend != "kernel":
            raise RuntimeError(
                "SharQ linear execution requires the real kernel-backed prepare path."
            )

        sharq_ops = load_sharq_ops()
        output_scale = float(
            (prepared.scale * layer.weight_scale).detach().float().cpu().item()
        )
        y_sparse = sharq_ops.sparse_matmul(
            prepared.a_comp,
            layer.qweight,
            prepared.e,
            prepared.sfa_sparse,
            layer.sfw_sparse,
            x_2d.shape[0],
            layer.output_size_per_partition,
            layer.input_size_per_partition,
            alpha=output_scale,
        )
        if self.quant_config.extra_fusion:
            y_2d = sharq_ops.matmul_accum(
                prepared.q_res,
                layer.qweight,
                prepared.sf_res,
                layer.sfw_dense,
                output_scale,
                y_sparse,
                1.0,
            )
        else:
            y_res = sharq_ops.matmul(
                prepared.q_res,
                layer.qweight,
                prepared.sf_res,
                layer.sfw_dense,
                output_scale,
            )
            y_2d = y_sparse + y_res

        if bias is not None:
            y_2d = y_2d + bias
        return y_2d.reshape(*input_shape[:-1], layer.output_size_per_partition)
