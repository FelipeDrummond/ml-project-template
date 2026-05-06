"""LR schedules.

Currently implements linear-warmup-then-cosine, the standard for
supervised / behavior-cloning workloads. RL training loops (PPO, SAC)
typically use constant LR by convention — leave `cfg.trainer.scheduler=null`.

Refs:
- Linear warmup: Goyal et al. 2017 (arXiv:1706.02677).
- Cosine decay: Loshchilov & Hutter 2017 (arXiv:1608.03983).
- Why warmup: Liu et al. 2020, RAdam (arXiv:1908.03265) — Adam's
  second-moment estimate is unreliable in the first ~hundred steps.
"""

from __future__ import annotations

import logging
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

logger = logging.getLogger(__name__)

VALID_SCHEDULERS = frozenset({"linear_warmup_cosine"})


def resolve_warmup_steps(
    requested: int | None, total_optim_steps: int, frac: float = 0.1, cap: int = 1000
) -> int:
    """`null` -> auto: min(frac * total, cap). Otherwise pass-through."""
    if requested is not None:
        return requested
    return min(int(frac * total_optim_steps), cap)


def build_scheduler(
    name: str | None,
    optim: Optimizer,
    total_optim_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LRScheduler | None:
    """Construct the LR scheduler. Returns `None` for constant LR."""
    if name is None:
        return None
    if name not in VALID_SCHEDULERS:
        raise ValueError(f"Unknown scheduler {name!r}. Must be one of {sorted(VALID_SCHEDULERS)}.")
    if total_optim_steps <= 0:
        raise ValueError(
            f"total_optim_steps must be > 0, got {total_optim_steps}. "
            f"Check that grad_accum_steps <= len(train_loader)."
        )
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0,1], got {min_lr_ratio}")
    if warmup_steps < 0 or warmup_steps > total_optim_steps:
        raise ValueError(f"warmup_steps={warmup_steps} out of range [0,{total_optim_steps}]")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear ramp from 0 to 1 over warmup_steps.
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay from 1 → min_lr_ratio over the remainder.
        progress = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    logger.info(
        "Built %s scheduler: total_optim_steps=%d warmup_steps=%d min_lr_ratio=%.3f",
        name,
        total_optim_steps,
        warmup_steps,
        min_lr_ratio,
    )
    return LambdaLR(optim, lr_lambda=lr_lambda)


def current_lr(optim: Optimizer | torch.optim.Optimizer) -> float:
    """First param-group LR; sufficient for single-group AdamW."""
    return float(optim.param_groups[0]["lr"])
