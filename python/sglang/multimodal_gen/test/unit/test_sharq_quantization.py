import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.quantization.configs import sharq_config
from sglang.multimodal_gen.runtime.layers.quantization.configs.sharq_config import (
    SharQConfig,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.sharq_linear import (
    SharQLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.sharq_ops import (
    dense_scale_buffer_numel,
    sparse_scale_buffer_numel,
)
from sglang.multimodal_gen.runtime.layers.quantization.sharq_prepare import (
    SharQPreparedActivation,
    prepare_sharq_activation,
    prepare_sharq_activation_after_rmsnorm,
)
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability


def _sharq_payload() -> dict:
    return {
        "quant_method": "sharq",
        "format_version": 1,
        "target_model": "WanTransformer3DModel",
        "target_pipeline": "Wan2.2-T2V-A14B",
        "extra_fusion": True,
        "tp_supported": False,
        "fused_modules": [],
        "weight_format": "sharq_w32_shared_nvfp4",
    }


class TestSharQConfig(unittest.TestCase):
    def test_from_pretrained_reads_quantization_config_from_config_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            qdir = Path(tmpdir) / "wan22-sharq"
            qdir.mkdir()
            (qdir / "config.json").write_text(
                json.dumps(
                    {
                        "_class_name": "WanTransformer3DModel",
                        "quantization_config": _sharq_payload(),
                    }
                )
            )

            cfg = SharQConfig.from_pretrained(str(qdir))

        self.assertIsInstance(cfg, SharQConfig)
        assert cfg is not None
        self.assertEqual(cfg.weight_format, "sharq_w32_shared_nvfp4")
        self.assertEqual(cfg.fused_modules, [])
        self.assertEqual(cfg.transformer_weights_path, str(qdir))

    def test_sharq_validate_runtime_rejects_tp_gt_1(self):
        cfg = SharQConfig.from_config(_sharq_payload())

        with (
            patch.object(sharq_config.current_platform, "is_cuda", return_value=True),
            patch.object(
                sharq_config.current_platform,
                "get_device_capability",
                return_value=DeviceCapability(12, 0),
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.quantization.configs.sharq_config.load_sharq_ops",
                return_value=object(),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "tp_size == 1"):
                cfg.validate_runtime(
                    model_cls_name="WanTransformer3DModel",
                    pipeline_name="Wan2_2_T2V_A14B_Config",
                    tp_size=2,
                )

    def test_sharq_validate_runtime_rejects_fused_wan_checkpoint(self):
        payload = _sharq_payload()
        payload["fused_modules"] = ["attn1.to_qkv", "attn2.to_kv"]
        cfg = SharQConfig.from_config(payload)

        with (
            patch.object(sharq_config.current_platform, "is_cuda", return_value=True),
            patch.object(
                sharq_config.current_platform,
                "get_device_capability",
                return_value=DeviceCapability(12, 0),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "fused_modules=\\["):
                cfg.validate_runtime(
                    model_cls_name="WanTransformer3DModel",
                    pipeline_name="Wan2_2_T2V_A14B_Config",
                    tp_size=1,
                )

    def test_sharq_config_skips_ignored_linear_prefixes(self):
        payload = _sharq_payload()
        payload["modules_to_not_convert"] = [
            "blocks.0.attn2.to_k",
            "time_modulation.linear",
        ]
        cfg = SharQConfig.from_config(payload)

        skipped_layer = ReplicatedLinear(
            128,
            128,
            quant_config=cfg,
            prefix="blocks.0.attn2.to_k",
        )
        quantized_layer = ReplicatedLinear(
            128,
            128,
            quant_config=cfg,
            prefix="blocks.0.attn1.to_q",
        )

        self.assertIsInstance(skipped_layer.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(quantized_layer.quant_method, SharQLinearMethod)


class TestSharQPrepare(unittest.TestCase):
    def test_prepare_sharq_activation_reference_backend(self):
        x = torch.linspace(-1.0, 1.0, steps=256, dtype=torch.float32).reshape(2, 128)

        prepared = prepare_sharq_activation(
            x,
            out_features=128,
            backend="reference",
        )

        self.assertEqual(prepared.backend, "reference")
        self.assertEqual(prepared.x_sparse_q.shape, x.shape)
        self.assertEqual(prepared.x_res_q.shape, x.shape)
        self.assertGreater(prepared.scale.item(), 0.0)

    def test_prepare_sharq_activation_after_rmsnorm_reference_backend(self):
        x = torch.randn(2, 128, dtype=torch.float32)
        rmsnorm_weight = torch.ones(128, dtype=torch.bfloat16)

        prepared = prepare_sharq_activation_after_rmsnorm(
            x,
            rmsnorm_weight,
            rmsnorm_eps=1e-6,
            out_features=128,
            backend="reference",
        )

        self.assertEqual(prepared.backend, "reference")
        self.assertEqual(prepared.x_sparse_q.shape, x.shape)
        self.assertEqual(prepared.x_res_q.shape, x.shape)
        self.assertGreater(prepared.scale.item(), 0.0)


class TestSharQLinearMethod(unittest.TestCase):
    def test_create_weights_declares_missing_param_init(self):
        layer = torch.nn.Module()
        method = SharQLinearMethod(SharQConfig())

        method.create_weights(
            layer,
            input_size_per_partition=128,
            output_partition_sizes=[64],
            input_size=128,
            output_size=64,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(layer.qweight.missing_param_init, "zeros")
        self.assertEqual(layer.sfw_sparse.missing_param_init, "zeros")
        self.assertEqual(layer.sfw_dense.missing_param_init, "zeros")
        self.assertEqual(layer.weight_scale.missing_param_init, "ones")

    def test_create_weights_matches_exported_sparse_and_dense_shapes(self):
        layer = torch.nn.Module()
        method = SharQLinearMethod(SharQConfig())

        method.create_weights(
            layer,
            input_size_per_partition=5120,
            output_partition_sizes=[5120],
            input_size=5120,
            output_size=5120,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(
            layer.sfw_sparse.numel(), sparse_scale_buffer_numel(5120, 5120)
        )
        self.assertEqual(
            layer.sfw_dense.numel(), dense_scale_buffer_numel(5120, 5120)
        )
        self.assertLess(layer.sfw_sparse.numel(), layer.sfw_dense.numel())

    def test_sparse_scale_shape_pads_small_row_counts_to_one_block(self):
        layer = torch.nn.Module()
        method = SharQLinearMethod(SharQConfig())

        method.create_weights(
            layer,
            input_size_per_partition=5120,
            output_partition_sizes=[64],
            input_size=5120,
            output_size=64,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(layer.sfw_sparse.numel(), sparse_scale_buffer_numel(64, 5120))
        self.assertEqual(layer.sfw_dense.numel(), dense_scale_buffer_numel(64, 5120))
        self.assertEqual(layer.sfw_sparse.numel(), 20480)

    def test_apply_uses_kernel_prepare_contract(self):
        class FakeSharQOps:
            def __init__(self):
                self.last_alpha = None

            def sparse_matmul(self, *_args, alpha):
                self.last_alpha = alpha
                return torch.full((2, 4), 1.0, dtype=torch.bfloat16)

            def matmul_accum(self, *_args):
                y_sparse = _args[-2]
                return y_sparse + 1.0

        fake_ops = FakeSharQOps()
        prepared = SharQPreparedActivation(
            backend="kernel",
            scale=torch.tensor(2.0, dtype=torch.float32),
            a_comp=torch.zeros(1, dtype=torch.uint8),
            e=torch.zeros(1, dtype=torch.uint8),
            sfa_sparse=torch.zeros(1, dtype=torch.uint8),
            q_res=torch.zeros(2, 64, dtype=torch.uint8),
            sf_res=torch.zeros(1, dtype=torch.uint8),
        )

        layer = torch.nn.Module()
        layer.output_size_per_partition = 4
        layer.input_size_per_partition = 128
        layer.qweight = torch.nn.Parameter(
            torch.zeros(4, 64, dtype=torch.uint8), requires_grad=False
        )
        layer.sfw_sparse = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.uint8), requires_grad=False
        )
        layer.sfw_dense = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.uint8), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.tensor([0.5], dtype=torch.float32), requires_grad=False
        )

        method = SharQLinearMethod(SharQConfig())
        x = torch.randn(2, 128, dtype=torch.bfloat16)

        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.quantization.sharq_linear.prepare_sharq_activation",
                return_value=prepared,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.quantization.sharq_linear.load_sharq_ops",
                return_value=fake_ops,
            ),
        ):
            out = method.apply(layer, x)

        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(fake_ops.last_alpha, 1.0)
        torch.testing.assert_close(out, torch.full((2, 4), 2.0, dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
