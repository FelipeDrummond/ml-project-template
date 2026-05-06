"""Performance / cost-efficiency knobs for cloud GPU training.

Observability note on `gpu_utilization` (returned by nvidia-smi /
pynvml): it's "% of time at least one kernel was running", **not**
saturation. A model running tiny kernels back-to-back shows 100% util
while doing almost nothing useful. Treat it as a coarse "is the GPU
idle right now" signal, not a "how efficient is the kernel" signal.
The proper saturation metric is MFU (Model FLOPs Utilization) — we
don't ship it because the FLOP counter API is fragile and the user
explicitly chose util% for simplicity.


Each helper here is a small, well-isolated lever that improves GPU
throughput or prevents wasted compute. They're composed into the training
loop based on `TrainerConfig` flags so a run can opt in/out from a config
without code changes.

Reference points for the defaults used elsewhere:
- TF32: only meaningful for fp32 ops outside autocast (LayerNorm, optimizer
  math). Free on Ampere+, no-op on older.
- Fused AdamW: ~10-20% faster optimizer step on CUDA. Fallback to default
  (which is `foreach=True` on PyTorch 2.0+) on non-CUDA hosts.
- Grad clipping: a regularization / training-stability technique
  (Pascanu et al. 2013, arXiv:1211.5063), NOT a NaN preventer. NaN guard
  is a separate, cheap check.
"""

from __future__ import annotations

import logging

import torch

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


def enable_tf32(enabled: bool = True) -> None:
    """Toggle TF32 for fp32 matmul on Ampere+ GPUs.

    Free ~30% speedup for fp32 ops; no-op on older GPUs and on MPS/CPU.
    Mostly matters for ops *outside* autocast (LayerNorm, optimizer math) —
    redundant for matmuls that are already in bf16/fp16.
    """
    precision = "high" if enabled else "highest"
    torch.set_float32_matmul_precision(precision)
    logger.info("torch.set_float32_matmul_precision(%r)", precision)


def use_fused_adamw() -> bool:
    """Whether `fused=True` is safe to pass to `torch.optim.AdamW`.

    Fused AdamW only works on CUDA tensors. On Mac (MPS) and CPU we fall
    back to the default implementation (which is `foreach=True` on
    PyTorch 2.0+).
    """
    return torch.cuda.is_available()


def assert_finite_loss(loss: torch.Tensor, step: int) -> None:
    """Cheap NaN/Inf guard. Raises so a divergent run dies in seconds."""
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Loss is non-finite at step {step}: {loss.item()!r}. "
            f"Set trainer.detect_anomaly=true to locate the offending op."
        )


def gpu_memory_snapshot() -> dict[str, float]:
    """Return current and peak CUDA memory in MiB, plus current util%.

    Empty dict on non-CUDA hosts. `peak` is `max_memory_allocated()`
    since the last `reset_peak_memory_stats()` call — so log per-epoch
    and reset, otherwise peak monotonically grows for the whole run.
    """
    if not torch.cuda.is_available():
        return {}

    mib = 1024 * 1024
    snap: dict[str, float] = {
        "gpu/mem_alloc_mib": torch.cuda.memory_allocated() / mib,
        "gpu/mem_reserved_mib": torch.cuda.memory_reserved() / mib,
        "gpu/mem_peak_mib": torch.cuda.max_memory_allocated() / mib,
    }

    # pynvml: util%. Best-effort — missing libnvidia-ml.so on the host,
    # missing driver, or pynvml itself is fine. Just no util% logged.
    if _PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            snap["gpu/util_pct"] = float(util.gpu)
            snap["gpu/mem_util_pct"] = float(util.memory)
        except Exception:
            pass

    return snap


def reset_peak_memory_stats() -> None:
    """Reset CUDA peak-memory tracker. Call once per epoch."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
