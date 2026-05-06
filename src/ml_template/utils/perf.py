"""Performance / cost-efficiency knobs for cloud GPU training.

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
