from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.layers.quantization.sharq_ops import (
    global_nvfp4_scale,
    load_sharq_ops,
)

DEFAULT_TARGET_MODEL = "WanTransformer3DModel"
DEFAULT_TARGET_PIPELINE = "Wan2.2-T2V-A14B"
DEFAULT_COMPONENTS = ("transformer", "transformer_2")
QuantizeWeightFn = Callable[
    [torch.Tensor, torch.Tensor | None], dict[str, torch.Tensor]
]


def _load_transformer_state_dict(input_transformer: Path) -> dict[str, torch.Tensor]:
    safetensors_files = sorted(input_transformer.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(
            f"No safetensors files found under transformer directory: {input_transformer}"
        )

    state_dict: dict[str, torch.Tensor] = {}
    for safetensors_file in safetensors_files:
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in state_dict:
                    raise ValueError(
                        f"Duplicate tensor key {key!r} found while loading {input_transformer}"
                    )
                state_dict[key] = f.get_tensor(key)
    return state_dict


def _build_quantization_config(
    target_pipeline: str = DEFAULT_TARGET_PIPELINE,
) -> dict[str, object]:
    return {
        "quant_method": "sharq",
        "format_version": 1,
        "target_model": DEFAULT_TARGET_MODEL,
        "target_pipeline": target_pipeline,
        "extra_fusion": True,
        "tp_supported": False,
        "fused_modules": [],
        "weight_format": "sharq_w32_shared_nvfp4",
    }


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def _quantize_weight_with_sharq_ops(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(
            f"SharQ export only supports 2D linear weights, got shape {tuple(weight.shape)}"
        )
    if weight.shape[1] % 32 != 0:
        raise ValueError(
            "SharQ requires input features to be a multiple of 32 during export, "
            f"got weight shape {tuple(weight.shape)}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SharQ export requires CUDA because quantize_w32_shared runs in the "
            "external sharq_ops extension."
        )

    sharq_ops = load_sharq_ops()
    weight_bf16 = weight.to(device="cuda", dtype=torch.bfloat16).contiguous()
    weight_scale = global_nvfp4_scale(weight_bf16)
    qweight, sfw_sparse, sfw_dense = sharq_ops.quantize_w32_shared(
        (weight_bf16 / weight_scale).to(torch.bfloat16)
    )

    result = {
        "qweight": _cpu_tensor(qweight),
        "sfw_sparse": _cpu_tensor(sfw_sparse),
        "sfw_dense": _cpu_tensor(sfw_dense),
        "weight_scale": weight_scale.detach().cpu().reshape(1).to(torch.float32),
    }
    if bias is not None:
        result["bias"] = _cpu_tensor(bias)
    return result


def convert_wan_state_dict_to_sharq(
    *,
    state_dict: dict[str, torch.Tensor],
    quantize_weight_fn: QuantizeWeightFn | None = None,
) -> dict[str, torch.Tensor]:
    quantize_weight_fn = quantize_weight_fn or _quantize_weight_with_sharq_ops
    exported: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    for key in sorted(state_dict):
        if key in consumed:
            continue

        tensor = state_dict[key]
        if key.endswith(".weight") and tensor.ndim == 2:
            module_prefix = key[: -len(".weight")]
            bias_key = f"{module_prefix}.bias"
            bias = state_dict.get(bias_key)
            for suffix, quantized_tensor in quantize_weight_fn(
                tensor.to(torch.bfloat16).contiguous(),
                bias.contiguous() if bias is not None else None,
            ).items():
                exported[f"{module_prefix}.{suffix}"] = _cpu_tensor(quantized_tensor)
            consumed.add(key)
            if bias is not None:
                consumed.add(bias_key)
            continue

        exported[key] = _cpu_tensor(tensor)

    return exported


def export_wan_to_sharq(
    *,
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
    target_pipeline: str = DEFAULT_TARGET_PIPELINE,
    config: dict | None = None,
    quantize_weight_fn: QuantizeWeightFn | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exported = convert_wan_state_dict_to_sharq(
        state_dict=state_dict,
        quantize_weight_fn=quantize_weight_fn,
    )
    save_file(
        exported,
        str(output_path / "diffusion_pytorch_model.safetensors"),
        metadata={"format": "pt"},
    )

    config_payload = dict(config or {})
    config_payload["quantization_config"] = _build_quantization_config(
        target_pipeline=target_pipeline
    )

    with open(output_path / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)
        f.write("\n")


def export_wan_transformer_dir_to_sharq(
    *,
    input_transformer: str,
    output_dir: str,
    target_pipeline: str = DEFAULT_TARGET_PIPELINE,
    quantize_weight_fn: QuantizeWeightFn | None = None,
) -> None:
    input_path = Path(input_transformer)
    if not input_path.is_dir():
        raise ValueError(
            f"Expected --input-transformer to be a directory, got {input_transformer}"
        )

    config_path = input_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing config.json under transformer directory: {input_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    export_wan_to_sharq(
        state_dict=_load_transformer_state_dict(input_path),
        output_dir=output_dir,
        target_pipeline=target_pipeline,
        config=config,
        quantize_weight_fn=quantize_weight_fn,
    )


def export_wan_model_to_sharq(
    *,
    input_model_path: str,
    output_dir: str,
    target_pipeline: str = DEFAULT_TARGET_PIPELINE,
    components: tuple[str, ...] = DEFAULT_COMPONENTS,
    quantize_weight_fn: QuantizeWeightFn | None = None,
) -> list[str]:
    input_path = Path(input_model_path)
    if not input_path.is_dir():
        raise ValueError(
            f"Expected --input-model-path to be a directory, got {input_model_path}"
        )

    exported_components: list[str] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for component in components:
        component_input_path = input_path / component
        if not component_input_path.is_dir():
            continue
        export_wan_transformer_dir_to_sharq(
            input_transformer=str(component_input_path),
            output_dir=str(output_path / component),
            target_pipeline=target_pipeline,
            quantize_weight_fn=quantize_weight_fn,
        )
        exported_components.append(component)

    if not exported_components:
        raise ValueError(
            f"No Wan transformer components found under {input_model_path}; "
            f"expected one of {components}."
        )
    return exported_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Wan Diffusers transformer weights to SharQ checkpoints."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-transformer",
        type=str,
        help="Path to a single source Diffusers transformer directory.",
    )
    input_group.add_argument(
        "--input-model-path",
        type=str,
        help=(
            "Path to the source Wan Diffusers model root. Matching transformer "
            "subdirectories such as transformer/ and transformer_2/ will each "
            "be exported into the output root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the exported SharQ checkpoint will be written.",
    )
    parser.add_argument(
        "--target-pipeline",
        type=str,
        default=DEFAULT_TARGET_PIPELINE,
        help="Pipeline identifier stored in quantization_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_model_path:
        export_wan_model_to_sharq(
            input_model_path=args.input_model_path,
            output_dir=args.output_dir,
            target_pipeline=args.target_pipeline,
        )
        return

    export_wan_transformer_dir_to_sharq(
        input_transformer=args.input_transformer,
        output_dir=args.output_dir,
        target_pipeline=args.target_pipeline,
    )


if __name__ == "__main__":
    main()
