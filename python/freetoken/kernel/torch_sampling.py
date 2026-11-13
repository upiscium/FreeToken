"""Pure-PyTorch sampling with the flashinfer.sampling / triton sampling API.

Added by TSUMGI (2026-11) for pre-Volta GPUs (sm_6x). The fused Triton samplers use
acq_rel atomics for a cross-CTA spin barrier, which PTX only provides on sm_70+.
These versions trade a little speed for portability: one sort per row per step.

``deterministic``, ``generator`` and ``check_nan`` are accepted for signature
compatibility. ``seed`` (with ``offset``) seeds a local generator so draws reproduce.
"""

from __future__ import annotations

import torch

_MIN_TEMPERATURE = 1e-5


def _rows(value, B: int, device, dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).reshape(-1).expand(B) if value.numel() == 1 \
            else value.to(device=device, dtype=dtype).reshape(B)
    return torch.full((B,), value, device=device, dtype=dtype)


def _generator(device, seed, offset):
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    s = int(seed.reshape(-1)[0].item()) if isinstance(seed, torch.Tensor) else int(seed)
    o = 0 if offset is None else (int(offset.reshape(-1)[0].item()) if isinstance(offset, torch.Tensor) else int(offset))
    gen.manual_seed((s * 1_000_003 + o) & 0x7FFF_FFFF_FFFF_FFFF)
    return gen


def softmax(logits, temperature=None, enable_pdl=None):
    logits = logits.float()
    if logits.size(0) == 0:
        return logits.clone()
    if temperature is None:
        return torch.softmax(logits, dim=-1)
    t = _rows(temperature, logits.size(0), logits.device, torch.float32).clamp_min(_MIN_TEMPERATURE)
    return torch.softmax(logits / t.unsqueeze(1), dim=-1)


def _renorm(probs: torch.Tensor) -> torch.Tensor:
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)


def _apply_top_k(probs: torch.Tensor, top_k) -> torch.Tensor:
    B, V = probs.shape
    k = _rows(top_k, B, probs.device, torch.int64)
    k = torch.where((k <= 0) | (k > V), torch.full_like(k, V), k)
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    thr = sorted_probs.gather(1, (k - 1).unsqueeze(1))
    return probs.masked_fill(probs < thr, 0.0)


def _apply_top_p(probs: torch.Tensor, top_p) -> torch.Tensor:
    B, _ = probs.shape
    p = _rows(top_p, B, probs.device, torch.float32)
    probs = _renorm(probs)
    sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
    before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    drop_sorted = before >= p.unsqueeze(1)
    drop_sorted[:, 0] = False  # always keep the most likely token
    drop = torch.zeros_like(drop_sorted).scatter(1, idx, drop_sorted)
    return probs.masked_fill(drop, 0.0)


def _draw(probs: torch.Tensor, seed, offset) -> torch.Tensor:
    if probs.size(0) == 0:
        return torch.empty(0, device=probs.device, dtype=torch.int32)
    gen = _generator(probs.device, seed, offset)
    return torch.multinomial(_renorm(probs), 1, generator=gen).squeeze(1).to(torch.int32)


def _select(probs, indices):
    probs = probs.float()
    return probs if indices is None else probs[indices].contiguous()


def _finish(out, indices, return_valid):
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_renorm_probs(probs, top_k):
    probs = probs.float()
    return _renorm(_apply_top_k(probs, top_k)) if probs.size(0) else probs.clone()


def top_p_renorm_probs(probs, top_p):
    probs = probs.float()
    return _renorm(_apply_top_p(probs, top_p)) if probs.size(0) else probs.clone()


def sampling_from_probs(probs, indices=None, deterministic=True, generator=None,
                        check_nan=False, seed=None, offset=None, return_valid=False):
    return _finish(_draw(_select(probs, indices), seed, offset), indices, return_valid)


def top_k_sampling_from_probs(probs, top_k, indices=None, deterministic=True, generator=None,
                              check_nan=False, seed=None, offset=None, return_valid=False):
    src = _select(probs, indices)
    out = _draw(_apply_top_k(src, top_k), seed, offset) if src.size(0) else _draw(src, seed, offset)
    return _finish(out, indices, return_valid)


def top_p_sampling_from_probs(probs, top_p, indices=None, deterministic=True, generator=None,
                              check_nan=False, seed=None, offset=None, return_valid=False):
    src = _select(probs, indices)
    out = _draw(_apply_top_p(src, top_p), seed, offset) if src.size(0) else _draw(src, seed, offset)
    return _finish(out, indices, return_valid)


def top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices=None,
                                    filter_apply_order="top_k_first", deterministic=True,
                                    generator=None, check_nan=False, seed=None, offset=None,
                                    return_valid=False):
    src = _select(probs, indices)
    if src.size(0):
        src = _apply_top_p(_apply_top_k(src, top_k), top_p)
    return _finish(_draw(src, seed, offset), indices, return_valid)


__all__ = [
    "softmax", "top_k_renorm_probs", "top_p_renorm_probs",
    "sampling_from_probs", "top_k_sampling_from_probs",
    "top_p_sampling_from_probs", "top_k_top_p_sampling_from_probs",
]
