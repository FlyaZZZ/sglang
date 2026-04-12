from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import torch

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.sharq_ops import (
    is_sharq_available as _is_sharq_available_impl,
    load_sharq_ops,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


@lru_cache(maxsize=1)
def is_sharq_available() -> bool:
    return _is_sharq_available_impl()


@dataclass
class SharQConfig(QuantizationConfig):
    format_version: int = 1
    transformer_weights_path: Optional[str] = None
    target_model: str = "WanTransformer3DModel"
    target_pipeline: str = "Wan2.2-T2V-A14B"
    extra_fusion: bool = True
    tp_supported: bool = False
    fused_modules: list[str] = field(default_factory=list)
    weight_format: str = "sharq_w32_shared_nvfp4"

    def __post_init__(self) -> None:
        QuantizationConfig.__init__(self)
        self.fused_modules = list(self.fused_modules or [])

    @classmethod
    def get_name(cls) -> str:
        return "sharq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json", "quantization_config.json", "quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SharQConfig":
        return cls(
            format_version=int(config.get("format_version", 1)),
            transformer_weights_path=config.get("transformer_weights_path"),
            target_model=config.get("target_model", "WanTransformer3DModel"),
            target_pipeline=config.get("target_pipeline", "Wan2.2-T2V-A14B"),
            extra_fusion=bool(config.get("extra_fusion", True)),
            tp_supported=bool(config.get("tp_supported", False)),
            fused_modules=list(config.get("fused_modules") or []),
            weight_format=config.get("weight_format", "sharq_w32_shared_nvfp4"),
        )

    @classmethod
    def from_pretrained(cls, model_path: str) -> Optional["SharQConfig"]:
        for filename in cls.get_config_filenames():
            config_path = os.path.join(model_path, filename)
            if not os.path.exists(config_path):
                continue
            with open(config_path, "r") as f:
                payload = json.load(f)
            if filename == "config.json":
                payload = payload.get("quantization_config")
            if not isinstance(payload, dict):
                continue
            if payload.get("quant_method") != cls.get_name():
                continue
            config = cls.from_config(payload)
            if config.transformer_weights_path is None:
                config.transformer_weights_path = model_path
            return config
        return None

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    def _pipeline_matches(self, pipeline_name: str | None) -> bool:
        if not pipeline_name:
            return False
        expected = self._normalize_name(self.target_pipeline)
        actual = self._normalize_name(pipeline_name)
        return expected in actual or actual in expected

    def validate_runtime(
        self,
        *,
        model_cls_name: str,
        pipeline_name: str | None,
        tp_size: int,
    ) -> None:
        if not current_platform.is_cuda():
            raise ValueError("SharQ requires a CUDA runtime.")

        device_capability = current_platform.get_device_capability()
        if device_capability is None or device_capability.to_int() < self.get_min_capability():
            got = (
                device_capability.as_version_str()
                if device_capability is not None
                else "unknown"
            )
            raise ValueError(
                "SharQ requires Blackwell-class CUDA devices (SM120+), "
                f"but got compute capability {got}."
            )

        if tp_size != 1:
            raise ValueError(
                f"SharQ milestone one requires tp_size == 1, got tp_size={tp_size}."
            )

        if self.fused_modules:
            raise ValueError(
                "This SharQ runtime expects split Wan projection checkpoints with "
                f"fused_modules=[], but got fused_modules={self.fused_modules}. "
                "Re-export both transformer components with the current "
                "convert_wan_to_sharq tool."
            )

        try:
            load_sharq_ops()
        except Exception as exc:
            raise ValueError(
                "SharQ is enabled but `sharq_ops` could not be imported. "
                "Build the SharQ extension or set SGLANG_SHARQ_OPS_PATH."
            ) from exc

        if model_cls_name != self.target_model:
            raise ValueError(
                f"SharQ checkpoint target_model={self.target_model} does not "
                f"match model class {model_cls_name}."
            )

        if not self._pipeline_matches(pipeline_name):
            raise ValueError(
                f"SharQ checkpoint target_pipeline={self.target_pipeline} does not "
                f"match served pipeline {pipeline_name}."
            )

    def is_fused_module_enabled(self, module_name: str) -> bool:
        normalized = module_name.lower()
        return any(normalized.endswith(item.lower()) for item in self.fused_modules)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase
        from sglang.multimodal_gen.runtime.layers.quantization.sharq_linear import (
            SharQLinearMethod,
        )

        if isinstance(layer, LinearBase):
            return SharQLinearMethod(self)
        return None
