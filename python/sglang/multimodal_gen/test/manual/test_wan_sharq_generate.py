import os

import pytest


@pytest.mark.skipif(
    os.getenv("SGLANG_RUN_SHARQ_SMOKE") != "1",
    reason="SharQ smoke test requires a Blackwell GPU and built sharq_ops.",
)
def test_wan22_sharq_smoke():
    from sglang.multimodal_gen.runtime.layers.quantization.sharq_ops import (
        load_sharq_ops,
    )

    load_sharq_ops()
