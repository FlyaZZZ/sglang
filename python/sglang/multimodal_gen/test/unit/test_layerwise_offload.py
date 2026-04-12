import unittest

import torch

from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
    _align_numel_offset,
)


class TestLayerwiseOffloadAlignment(unittest.TestCase):
    def test_align_numel_offset_preserves_32_byte_alignment(self):
        cases = [
            (0, torch.float32, 0),
            (1, torch.float32, 8),
            (9, torch.float32, 16),
            (1, torch.bfloat16, 16),
            (17, torch.bfloat16, 32),
            (1, torch.uint8, 32),
            (33, torch.uint8, 64),
        ]

        for offset, dtype, expected in cases:
            with self.subTest(offset=offset, dtype=dtype):
                aligned = _align_numel_offset(offset, dtype)
                self.assertEqual(aligned, expected)
                self.assertEqual(
                    (aligned * torch.tensor([], dtype=dtype).element_size()) % 32,
                    0,
                )


if __name__ == "__main__":
    unittest.main()
