"""Helpers and adapters for transformer quantized checkpoint loading.

This module keeps format-specific loading quirks out of `TransformerLoader`.
The loader should stay focused on the generic load flow, while special cases
such as Nunchaku validation, NVFP4 fallback adjustments, and post-load patching
are handled here behind a small helper/adapter layer.
"""

import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
    _patch_nunchaku_scales,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.sharq_config import (
    SharQConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
    maybe_download_model,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    build_nvfp4_config_from_safetensors_list,
    get_metadata_from_safetensors_file,
    get_quant_config,
    get_quant_config_from_safetensors_metadata,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE
from sglang.srt.layers.quantization import QuantizationConfig

logger = init_logger(__name__)

PostLoadHook = Callable[[nn.Module], None]


@dataclass
class TransformerQuantLoadSpec:
    """Resolved loading plan for a transformer checkpoint."""

    safetensors_list: list[str]
    quant_config: Optional[QuantizationConfig]
    nunchaku_config: Optional[NunchakuConfig]
    param_dtype: Optional[torch.dtype]
    post_load_hooks: list[PostLoadHook] = field(default_factory=list)

    @property
    def runtime_quant_config(self) -> Optional[object]:
        if self.quant_config is not None:
            return self.quant_config
        return self.nunchaku_config


class _TransformerQuantAdapter:
    def prepare(self) -> None:
        """initialize"""
        pass

    def get_post_load_hooks(self) -> list[PostLoadHook]:
        """post - fsdp load - hook"""
        return []


class _NunchakuQuantAdapter(_TransformerQuantAdapter):
    """Adapter for Nunchaku checkpoints"""

    def __init__(
        self,
        *,
        nunchaku_config: NunchakuConfig,
        model_cls: type[nn.Module],
        safetensors_list: list[str],
    ) -> None:
        self.nunchaku_config = nunchaku_config
        self.model_cls = model_cls
        self.safetensors_list = safetensors_list

    @staticmethod
    def _validate_nunchaku_checkpoint_matches_model(
        nunchaku_config: NunchakuConfig, model_cls: type[nn.Module]
    ) -> None:
        metadata = get_metadata_from_safetensors_file(
            nunchaku_config.transformer_weights_path
        )
        original_dit_cls_name = json.loads(metadata.get("config"))["_class_name"]
        specified_dit_cls_name = str(model_cls.__name__)
        if original_dit_cls_name != specified_dit_cls_name:
            raise Exception(
                f"Class name of DiT specified in nunchaku transformer_weights_path: "
                f"{original_dit_cls_name} does not match that of specified DiT name: "
                f"{specified_dit_cls_name}"
            )

    def prepare(self) -> None:
        self.nunchaku_config.model_cls = self.model_cls
        _NunchakuQuantAdapter._validate_nunchaku_checkpoint_matches_model(
            nunchaku_config=self.nunchaku_config,
            model_cls=self.model_cls,
        )

    def get_post_load_hooks(self) -> list[PostLoadHook]:
        return [partial(_patch_nunchaku_scales, safetensors_list=self.safetensors_list)]


class _SharQQuantAdapter(_TransformerQuantAdapter):
    """Adapter for SharQ override checkpoints."""

    def __init__(
        self,
        *,
        sharq_config: SharQConfig,
        server_args: ServerArgs,
        model_cls: type[nn.Module],
    ) -> None:
        self.sharq_config = sharq_config
        self.server_args = server_args
        self.model_cls = model_cls

    def prepare(self) -> None:
        pipeline_name = (
            self.server_args.model_id
            or os.path.basename(self.server_args.model_path or "")
            or self.server_args.pipeline_config.__class__.__name__
        )
        self.sharq_config.validate_runtime(
            model_cls_name=self.model_cls.__name__,
            pipeline_name=pipeline_name,
            tp_size=self.server_args.tp_size,
        )
        logger.info(
            "Resolved SharQ quantization for %s with override weights at %s",
            pipeline_name,
            self.server_args.transformer_weights_path,
        )


class _Flux2Nvfp4FallbackAdapter(_TransformerQuantAdapter):
    """Adapter for black-forest-labs/FLUX.2-dev-NVFP4"""

    def __init__(
        self,
        *,
        cls_name: str,
        server_args: ServerArgs,
        quant_config: Optional[QuantizationConfig],
    ) -> None:
        self.cls_name = cls_name
        self.server_args = server_args
        self.quant_config = quant_config

    @staticmethod
    def _maybe_adjust_flux2_nvfp4_fallback_defaults(
        cls_name: str,
        server_args: ServerArgs,
        quant_config: Optional[QuantizationConfig],
    ) -> None:
        if cls_name != "Flux2Transformer2DModel" or quant_config is None:
            return

        quant_name_getter = getattr(type(quant_config), "get_name", None)
        quant_name = quant_name_getter() if callable(quant_name_getter) else None
        if quant_name != "modelopt_fp4":
            return

        weights_path = os.path.basename(server_args.transformer_weights_path or "")
        if not weights_path.endswith("-mixed.safetensors") or server_args.tp_size <= 1:
            return

        if server_args.dit_cpu_offload or server_args.text_encoder_cpu_offload:
            server_args.dit_cpu_offload = False
            server_args.text_encoder_cpu_offload = False
            logger.warning(
                "FLUX.2 mixed NVFP4 is using the ModelOpt FP4 path with tp_size=%d; "
                "disabling dit/text-encoder CPU offload to avoid TP all-gather "
                "launch failures. Override the offload flags explicitly if you need "
                "the old behavior.",
                server_args.tp_size,
            )

    def prepare(self) -> None:
        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name=self.cls_name,
            server_args=self.server_args,
            quant_config=self.quant_config,
        )


def resolve_transformer_override_path(
    server_args: ServerArgs, component_model_path: str
) -> str | None:
    """Resolve the effective override path for the current transformer component.

    If ``--transformer-weights-path`` points at a root directory that contains
    component subdirectories such as ``transformer/`` and ``transformer_2/``,
    prefer the subdirectory matching the component currently being loaded.
    Otherwise, keep the original path for single-component checkpoints.
    """
    quantized_path = server_args.transformer_weights_path
    if not quantized_path:
        return None

    quantized_path = maybe_download_model(quantized_path)
    if not os.path.isdir(quantized_path):
        return quantized_path

    component_dir_name = os.path.basename(os.path.normpath(component_model_path))
    component_override_path = os.path.join(quantized_path, component_dir_name)
    if os.path.isdir(component_override_path):
        return component_override_path
    return quantized_path


def resolve_transformer_component_config(
    server_args: ServerArgs, component_model_path: str
) -> dict[str, Any]:
    """Load transformer config from the override dir when present, else base."""
    override_path = resolve_transformer_override_path(server_args, component_model_path)
    if override_path and os.path.isdir(override_path):
        return get_diffusers_component_config(component_path=override_path)
    return get_diffusers_component_config(component_path=component_model_path)


def resolve_transformer_safetensors_to_load(
    server_args: ServerArgs, component_model_path: str
) -> list[str]:
    """Resolve transformer weights from the base component path or an override."""
    quantized_path = resolve_transformer_override_path(server_args, component_model_path)

    if quantized_path:
        logger.info("using quantized transformer weights from: %s", quantized_path)
        if os.path.isfile(quantized_path) and quantized_path.endswith(".safetensors"):
            safetensors_list = [quantized_path]
        else:
            safetensors_list = _list_safetensors_files(quantized_path)
    else:
        safetensors_list = _list_safetensors_files(component_model_path)

    if not safetensors_list:
        raise ValueError(
            f"no safetensors files found in {quantized_path or component_model_path}"
        )

    return safetensors_list


def resolve_transformer_quant_load_spec(
    *,
    hf_config: dict,
    server_args: ServerArgs,
    safetensors_list: list[str],
    component_model_path: str,
    model_cls: type[nn.Module],
    cls_name: str,
) -> TransformerQuantLoadSpec:
    quant_config = _resolve_quant_config(
        hf_config=hf_config,
        server_args=server_args,
        safetensors_list=safetensors_list,
        component_model_path=component_model_path,
    )
    nunchaku_config = server_args.nunchaku_config

    # resolve target param dtype
    param_dtype = _resolve_target_param_dtype(
        quant_config=quant_config,
        nunchaku_config=nunchaku_config,
        server_args=server_args,
    )

    adapters = _build_transformer_quant_adapters(
        cls_name=cls_name,
        server_args=server_args,
        quant_config=quant_config,
        nunchaku_config=nunchaku_config,
        model_cls=model_cls,
        safetensors_list=safetensors_list,
    )
    for adapter in adapters:
        adapter.prepare()

    # collect post-load hooks from built adapters
    post_load_hooks: list[PostLoadHook] = []
    for adapter in adapters:
        post_load_hooks.extend(adapter.get_post_load_hooks())

    return TransformerQuantLoadSpec(
        safetensors_list=safetensors_list,
        quant_config=quant_config,
        nunchaku_config=nunchaku_config,
        param_dtype=param_dtype,
        post_load_hooks=post_load_hooks,
    )


def _build_transformer_quant_adapters(
    *,
    cls_name: str,
    server_args: ServerArgs,
    quant_config: Optional[QuantizationConfig],
    nunchaku_config: Optional[NunchakuConfig],
    model_cls: type[nn.Module],
    safetensors_list: list[str],
) -> list[_TransformerQuantAdapter]:
    adapters: list[_TransformerQuantAdapter] = [
        _Flux2Nvfp4FallbackAdapter(
            cls_name=cls_name,
            server_args=server_args,
            quant_config=quant_config,
        )
    ]
    if nunchaku_config is not None:
        adapters.append(
            _NunchakuQuantAdapter(
                nunchaku_config=nunchaku_config,
                model_cls=model_cls,
                safetensors_list=safetensors_list,
            )
        )
    if isinstance(quant_config, SharQConfig):
        adapters.append(
            _SharQQuantAdapter(
                sharq_config=quant_config,
                server_args=server_args,
                model_cls=model_cls,
            )
        )
    return adapters


def _resolve_quant_config(
    *,
    hf_config: dict,
    server_args: ServerArgs,
    safetensors_list: list[str],
    component_model_path: str,
) -> Optional[QuantizationConfig]:
    """
    Resolve quant config from the selected component config, checkpoint metadata,
    or format-specific fallback.
    Priority: override-dir config.json when present, otherwise base config.json
    -> safetensors metadata -> format-specific fallback.
    """
    quant_config = get_quant_config(hf_config, component_model_path)
    if quant_config is None and server_args.transformer_weights_path:
        for safetensors_file in safetensors_list:
            quant_config = get_quant_config_from_safetensors_metadata(safetensors_file)
            if quant_config is not None:
                return quant_config

        param_names_mapping_dict = (
            server_args.pipeline_config.dit_config.arch_config.param_names_mapping
        )
        quant_config = build_nvfp4_config_from_safetensors_list(
            safetensors_list, param_names_mapping_dict
        )
        if quant_config is not None:
            return quant_config

    return quant_config


def _resolve_target_param_dtype(
    *,
    quant_config: Optional[QuantizationConfig],
    nunchaku_config: Optional[NunchakuConfig],
    server_args: ServerArgs,
) -> Optional[torch.dtype]:
    if quant_config is not None or nunchaku_config is not None:
        return None
    return PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
