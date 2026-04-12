from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


_DEFAULT_SHARQ_REPO = Path("/data/SharQ")
_BUILD_SUBDIR_CANDIDATES = (
    "kernels/build_cmake_sm120a",
    "kernels/build",
)


def _candidate_search_paths() -> list[str]:
    candidates: list[str] = []

    for env_key in (
        "SGLANG_SHARQ_OPS_PATH",
        "SHARQ_OPS_PATH",
        "SHARQ_BUILD_DIR",
    ):
        env_value = os.environ.get(env_key)
        if env_value:
            candidates.append(env_value)

    sharq_repo = os.environ.get("SHARQ_REPO_PATH")
    if sharq_repo:
        repo_path = Path(sharq_repo)
        candidates.extend(str(repo_path / subdir) for subdir in _BUILD_SUBDIR_CANDIDATES)

    candidates.extend(
        str(_DEFAULT_SHARQ_REPO / subdir) for subdir in _BUILD_SUBDIR_CANDIDATES
    )
    return candidates


@lru_cache(maxsize=1)
def load_sharq_ops() -> ModuleType:
    """Import the external ``sharq_ops`` module with a few common search paths."""
    try:
        return importlib.import_module("sharq_ops")
    except ImportError:
        pass

    attempted_paths: list[str] = []
    for candidate in _candidate_search_paths():
        if not candidate or not os.path.isdir(candidate):
            continue
        attempted_paths.append(candidate)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            return importlib.import_module("sharq_ops")
        except ImportError:
            continue

    searched = ", ".join(attempted_paths) if attempted_paths else "<none>"
    raise ImportError(
        "Could not import `sharq_ops`. Install it into the current environment "
        "or point SGLang at a built SharQ extension with SGLANG_SHARQ_OPS_PATH "
        f"(searched: {searched})."
    )


def is_sharq_available() -> bool:
    try:
        load_sharq_ops()
        return True
    except Exception:
        return False


def global_nvfp4_scale(x):
    import torch

    return torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)


def scale_buffer_numel(num_rows: int, k_dim: int) -> int:
    return ((num_rows // 128) + 1) * 128 * k_dim // 16


__all__ = [
    "global_nvfp4_scale",
    "is_sharq_available",
    "load_sharq_ops",
    "scale_buffer_numel",
]
