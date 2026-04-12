import importlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open

from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoArchConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.sharq_config import (
    SharQConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.sharq_linear import (
    SharQLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.sharq_ops import (
    scale_buffer_numel,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_transformer_component_config,
    resolve_transformer_override_path,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.tools.convert_wan_to_sharq import (
    export_wan_model_to_sharq,
    export_wan_to_sharq,
)


class TestWanSharQStructure(unittest.TestCase):
    @staticmethod
    def _import_wanvideo_module():
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "wanvideo runtime imports require CUDA in this test environment"
            )
        return importlib.import_module(
            "sglang.multimodal_gen.runtime.models.dits.wanvideo"
        )

    def test_wan_param_mapping_keeps_split_attention_keys(self):
        mapping = WanVideoArchConfig().param_names_mapping

        self.assertTrue(any("attn1\\.to_q" in key for key in mapping))
        self.assertTrue(any("attn1\\.to_k" in key for key in mapping))
        self.assertTrue(any("attn1\\.to_v" in key for key in mapping))
        self.assertFalse(any("attn1\\.to_qkv" in key for key in mapping))
        self.assertFalse(any("attn2\\.to_kv" in key for key in mapping))

    def test_wan_sharq_keeps_split_qkv_and_kv(self):
        wanvideo = self._import_wanvideo_module()
        fake_group = types.SimpleNamespace(world_size=1, rank_in_group=0)

        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                return_value=fake_group,
            ),
            patch.object(wanvideo, "get_tp_world_size", return_value=1),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
                return_value=types.SimpleNamespace(attention_backend=None),
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                return_value=1,
            ),
        ):
            block = wanvideo.WanTransformerBlock(
                dim=128,
                ffn_dim=256,
                num_heads=8,
                qk_norm="rms_norm_across_heads",
                cross_attn_norm=True,
                supported_attention_backends=set(),
                quant_config=SharQConfig(),
            )

        self.assertIsNotNone(block.to_q)
        self.assertIsNotNone(block.to_k)
        self.assertIsNotNone(block.to_v)
        self.assertFalse(hasattr(block, "to_qkv"))
        self.assertIsNotNone(block.attn2.to_k)
        self.assertIsNotNone(block.attn2.to_v)
        self.assertFalse(hasattr(block.attn2, "to_kv"))

    def test_wan_condition_embedder_uses_sharq_linear_layers(self):
        wanvideo = self._import_wanvideo_module()
        fake_group = types.SimpleNamespace(world_size=1, rank_in_group=0)

        with patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
            return_value=fake_group,
        ):
            embedder = wanvideo.WanTimeTextImageEmbedding(
                dim=128,
                time_freq_dim=256,
                text_embed_dim=64,
                quant_config=SharQConfig(),
            )

        self.assertIsInstance(embedder.time_embedder.mlp.fc_in.quant_method, SharQLinearMethod)
        self.assertIsInstance(embedder.time_embedder.mlp.fc_out.quant_method, SharQLinearMethod)
        self.assertIsInstance(embedder.time_modulation.linear.quant_method, SharQLinearMethod)
        self.assertIsInstance(embedder.text_embedder.fc_in.quant_method, SharQLinearMethod)
        self.assertIsInstance(embedder.text_embedder.fc_out.quant_method, SharQLinearMethod)


class TestWanSharQExporter(unittest.TestCase):
    def test_export_wan_to_sharq_writes_split_attention_keys(self):
        state_dict = {
            "blocks.0.attn1.to_q.weight": torch.ones((128, 128), dtype=torch.bfloat16),
            "blocks.0.attn1.to_q.bias": torch.zeros((128,), dtype=torch.bfloat16),
            "blocks.0.attn1.to_k.weight": torch.full(
                (128, 128), 2.0, dtype=torch.bfloat16
            ),
            "blocks.0.attn1.to_k.bias": torch.ones((128,), dtype=torch.bfloat16),
            "blocks.0.attn1.to_v.weight": torch.full(
                (128, 128), 3.0, dtype=torch.bfloat16
            ),
            "blocks.0.attn1.to_v.bias": torch.full((128,), 2.0, dtype=torch.bfloat16),
            "blocks.0.attn2.to_k.weight": torch.ones((128, 128), dtype=torch.bfloat16),
            "blocks.0.attn2.to_v.weight": torch.full(
                (128, 128), 4.0, dtype=torch.bfloat16
            ),
            "blocks.0.ffn.net.0.proj.weight": torch.ones(
                (256, 128), dtype=torch.bfloat16
            ),
            "blocks.0.ffn.net.0.proj.bias": torch.zeros((256,), dtype=torch.bfloat16),
            "blocks.0.norm2.weight": torch.ones((128,), dtype=torch.bfloat16),
        }

        def fake_quantize(weight: torch.Tensor, bias: torch.Tensor | None):
            rows, cols = weight.shape
            scale_numel = scale_buffer_numel(rows, cols)
            payload = {
                "qweight": torch.zeros((rows, cols // 2), dtype=torch.uint8),
                "sfw_sparse": torch.zeros(scale_numel, dtype=torch.uint8),
                "sfw_dense": torch.zeros(scale_numel, dtype=torch.uint8),
                "weight_scale": torch.tensor([0.25], dtype=torch.float32),
            }
            if bias is not None:
                payload["bias"] = bias.clone()
            return payload

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            export_wan_to_sharq(
                state_dict=state_dict,
                output_dir=str(out_dir),
                target_pipeline="Wan2.2-T2V-A14B",
                config={"_class_name": "WanTransformer3DModel"},
                quantize_weight_fn=fake_quantize,
            )

            with open(out_dir / "config.json", "r") as f:
                config = json.load(f)
            self.assertEqual(config["_class_name"], "WanTransformer3DModel")
            self.assertEqual(config["quantization_config"]["quant_method"], "sharq")
            self.assertEqual(config["quantization_config"]["fused_modules"], [])

            with safe_open(
                str(out_dir / "diffusion_pytorch_model.safetensors"),
                framework="pt",
                device="cpu",
            ) as f:
                keys = set(f.keys())
                self.assertIn("blocks.0.attn1.to_q.qweight", keys)
                self.assertIn("blocks.0.attn1.to_k.qweight", keys)
                self.assertIn("blocks.0.attn1.to_v.qweight", keys)
                self.assertIn("blocks.0.attn1.to_q.bias", keys)
                self.assertIn("blocks.0.attn2.to_k.qweight", keys)
                self.assertIn("blocks.0.attn2.to_v.qweight", keys)
                self.assertIn("blocks.0.ffn.net.0.proj.qweight", keys)
                self.assertIn("blocks.0.norm2.weight", keys)
                self.assertEqual(
                    tuple(f.get_tensor("blocks.0.attn1.to_q.qweight").shape),
                    (128, 64),
                )

    def test_export_wan_model_to_sharq_writes_transformer_pair(self):
        def fake_quantize(weight: torch.Tensor, bias: torch.Tensor | None):
            rows, cols = weight.shape
            scale_numel = scale_buffer_numel(rows, cols)
            payload = {
                "qweight": torch.zeros((rows, cols // 2), dtype=torch.uint8),
                "sfw_sparse": torch.zeros(scale_numel, dtype=torch.uint8),
                "sfw_dense": torch.zeros(scale_numel, dtype=torch.uint8),
                "weight_scale": torch.tensor([0.5], dtype=torch.float32),
            }
            if bias is not None:
                payload["bias"] = bias.clone()
            return payload

        def write_component(component_dir: Path, weight_value: float):
            component_dir.mkdir(parents=True, exist_ok=True)
            with open(component_dir / "config.json", "w") as f:
                json.dump({"_class_name": "WanTransformer3DModel"}, f)
            state_dict = {
                "blocks.0.attn1.to_q.weight": torch.full(
                    (128, 128), weight_value, dtype=torch.bfloat16
                ),
                "blocks.0.attn1.to_k.weight": torch.full(
                    (128, 128), weight_value + 1, dtype=torch.bfloat16
                ),
                "blocks.0.attn1.to_v.weight": torch.full(
                    (128, 128), weight_value + 2, dtype=torch.bfloat16
                ),
            }
            save_path = component_dir / "diffusion_pytorch_model.safetensors"
            from safetensors.torch import save_file

            save_file(state_dict, str(save_path), metadata={"format": "pt"})

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "wan"
            out_dir = Path(tmpdir) / "wan-sharq"
            write_component(model_dir / "transformer", 1.0)
            write_component(model_dir / "transformer_2", 10.0)

            exported = export_wan_model_to_sharq(
                input_model_path=str(model_dir),
                output_dir=str(out_dir),
                target_pipeline="Wan2.2-T2V-A14B",
                quantize_weight_fn=fake_quantize,
            )

            self.assertEqual(exported, ["transformer", "transformer_2"])
            for component in exported:
                with open(out_dir / component / "config.json", "r") as f:
                    config = json.load(f)
                self.assertEqual(config["quantization_config"]["quant_method"], "sharq")
                self.assertFalse(
                    (out_dir / component / "quantization_config.json").exists()
                )
                self.assertTrue(
                    (out_dir / component / "diffusion_pytorch_model.safetensors").is_file()
                )


class TestWanSharQOverrideResolution(unittest.TestCase):
    def test_resolve_transformer_override_path_prefers_component_subdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            override_root = Path(tmpdir) / "wan-sharq"
            (override_root / "transformer").mkdir(parents=True)
            (override_root / "transformer_2").mkdir(parents=True)

            server_args = types.SimpleNamespace(
                transformer_weights_path=str(override_root)
            )

            transformer_path = resolve_transformer_override_path(
                server_args, "/base/model/transformer"
            )
            transformer_2_path = resolve_transformer_override_path(
                server_args, "/base/model/transformer_2"
            )

            self.assertEqual(transformer_path, str(override_root / "transformer"))
            self.assertEqual(transformer_2_path, str(override_root / "transformer_2"))

    def test_resolve_transformer_component_config_uses_override_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_root = Path(tmpdir) / "wan"
            base_transformer = base_root / "transformer"
            base_transformer.mkdir(parents=True)
            with open(base_transformer / "config.json", "w") as f:
                json.dump(
                    {
                        "_class_name": "WanTransformer3DModel",
                        "num_layers": 40,
                    },
                    f,
                )

            override_root = Path(tmpdir) / "wan-sharq"
            override_transformer = override_root / "transformer"
            override_transformer.mkdir(parents=True)
            with open(override_transformer / "config.json", "w") as f:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "sharq",
                            "target_model": "WanTransformer3DModel",
                        }
                    },
                    f,
                )

            server_args = types.SimpleNamespace(
                transformer_weights_path=str(override_root)
            )

            resolved = resolve_transformer_component_config(
                server_args, str(base_transformer)
            )

            self.assertEqual(
                resolved["quantization_config"]["quant_method"],
                "sharq",
            )
            self.assertNotIn("_class_name", resolved)
            self.assertNotIn("num_layers", resolved)

    def test_resolve_transformer_safetensors_to_load_uses_component_subdir(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            override_root = Path(tmpdir) / "wan-sharq"
            component_dir = override_root / "transformer_2"
            component_dir.mkdir(parents=True)
            save_file(
                {"blocks.0.norm2.weight": torch.ones(8, dtype=torch.float32)},
                str(component_dir / "diffusion_pytorch_model.safetensors"),
                metadata={"format": "pt"},
            )

            server_args = types.SimpleNamespace(
                transformer_weights_path=str(override_root)
            )
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/base/model/transformer_2"
            )

            self.assertEqual(
                resolved,
                [str(component_dir / "diffusion_pytorch_model.safetensors")],
            )


if __name__ == "__main__":
    unittest.main()
