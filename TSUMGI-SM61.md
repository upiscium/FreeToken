# TSUMGI pre-Turing (Pascal sm_61) support for FreeToken 0.2.2

This branch (`tsumgi/sm61-v0.2.2`) is based on upstream tag `v0.2.2` (`5a73685`).
It is maintained by the [TSUMGI](https://github.com/upiscium/TSUMGI) project so that
FreeToken can run on GTX 1080 Ti (compute capability 6.1). The upstream license
(Apache-2.0, see `LICENSE`) applies. Every changed file carries a "Modified by
TSUMGI" notice.

## Changes
1. `kernel/backend.py`: on GPUs below sm_75, report FlashInfer and sgl-kernel as
   unavailable. Their prebuilt kernels need sm_75+; FreeToken then takes its
   existing fallback paths. Set `FREETOKEN_FORCE_PRE_TURING=0/1` to override detection.
2. `kernel/csrc/include/freetoken/utils.cuh` and the JIT kernels `index.cu`,
   `store.cu`, `fast_index_copy.cuh`: `FREETOKEN_GRID_CONSTANT` drops
   `__grid_constant__` (compute_70+ only) on sm_6x device passes.
3. `kernel/triton/attention.py`: the extend/prefill tile choice falls back to
   smaller tiles when opt-in shared memory is small (48KB on Pascal). Upstream
   choices are unchanged when the first tile fits.

## Runtime requirements on sm_61 (not changed by this branch)
- PyTorch `2.14.1+cu126` (the default cu130 build has no sm_61 kernels), plus
  `nvidia-cuda-runtime==13.0.*` on `LD_LIBRARY_PATH` for the prebuilt `_pinned_tensor` extension.
- CUDA 12.6 toolkit (`cuda-nvcc-12-6`, `cuda-cudart-dev-12-6`, `libcublas-dev-12-6`),
  with `CUDA_HOME=/usr/local/cuda-12.6`. CUDA 13 cannot compile for sm_61.
- Python development headers (`python3.12-dev`) for Triton.
- `ft serve --attention-backend triton`.

## Verified
GTX 1080 Ti 11GB, driver 580.173.02, 16GB RAM: Qwen3-30B-A3B GGUF Q2_K with
offload MoE. Chat, SSE streaming, tool calls and a 5.3K-token prompt all worked
(decode ~55 tok/s, prefill ~500 tok/s). A 20-minute soak of 415 requests had 0 errors.
