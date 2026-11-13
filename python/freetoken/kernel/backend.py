"""Availability probes for the optional native kernel packages.

When flashinfer / sgl_kernel are installed the call-sites use their fused CUDA
ops; otherwise they fall back to the pure-Triton kernels in
``freetoken.kernel.triton``. ``find_spec`` only checks that the package is
importable (no import side effects), and the result is cached.
On ROCm, the optional-package probes return False so callers use the fallbacks.
"""
from __future__ import annotations

import functools
import os
import importlib.util

import torch


def _importable(name: str) -> bool:
    # find_spec normally returns None when a package is absent, but it can raise
    # (broken parent package, or a meta_path finder that blocks the name); treat
    # any failure as "not available" so callers cleanly fall back to triton.
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@functools.cache
def _pre_turing_gpu() -> bool:
    """Modified by TSUMGI (2026-11): prebuilt FlashInfer/sgl-kernel need sm_75+.

    On older GPUs report them as unavailable so FreeToken takes its existing
    non-FlashInfer/non-sgl-kernel code paths. FREETOKEN_FORCE_PRE_TURING=0/1
    overrides detection.
    """
    forced = os.environ.get("FREETOKEN_FORCE_PRE_TURING")
    if forced in ("0", "1"):
        return forced == "1"
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (7, 5)
    except Exception:
        return False


@functools.cache
def is_pre_volta_gpu() -> bool:
    """Modified by TSUMGI (2026-11): True below sm_70 (no acq_rel atomics in PTX)."""
    forced = os.environ.get("FREETOKEN_FORCE_PRE_VOLTA")
    if forced in ("0", "1"):
        return forced == "1"
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (7, 0)
    except Exception:
        return False


@functools.cache
def is_flashinfer_installed() -> bool:
    if is_rocm() or _pre_turing_gpu():
        return False
    return _importable("flashinfer")


@functools.cache
def is_sgl_kernel_installed() -> bool:
    if is_rocm() or _pre_turing_gpu():
        return False
    return _importable("sgl_kernel")


@functools.cache
def is_vllm_installed() -> bool:
    if is_rocm():
        return False
    return _importable("vllm")


@functools.cache
def device_capability() -> tuple[int, int]:
    """Compute capability of the current device as (major, minor); (0, 0) without CUDA."""
    if not torch.cuda.is_available():
        return (0, 0)
    major, minor = torch.cuda.get_device_capability()
    return (int(major), int(minor))


@functools.cache
def is_rocm() -> bool:
    """True when torch is built for ROCm (AMD GPU)."""
    import torch
    return getattr(torch.version, "hip", None) is not None


@functools.cache
def driver_cuda_version() -> int | None:
    """Max CUDA version the installed NVIDIA driver supports (``13000`` == CUDA 13.0),
    or None if undetermined. Driver-JIT kernels (PTX compiled at runtime, e.g.
    flashinfer's CuTe-DSL paths) are gated by this, not by any package's build-time
    toolkit version. Resolved through the ``_pinned_tensor`` extension's link-time
    cudart, so it works wherever the extension builds (including Windows) -- no dlopen
    by soname."""
    if is_rocm():
        return None
    try:
        from freetoken.kernel.pinned import _load_pinned_extension

        version = int(_load_pinned_extension().driver_cuda_version())
    except Exception:
        return None
    return version or None  # 0 == no driver installed
