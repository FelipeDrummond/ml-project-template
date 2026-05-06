"""Reproducibility helpers.

Note on CUDA non-determinism: even after `set_seed`, some CUDA kernels (e.g.
certain reductions, atomicAdd-based ops) are non-deterministic by default.
For full bit-exact reproducibility on CUDA, also set
`torch.use_deterministic_algorithms(True)` and the
`CUBLAS_WORKSPACE_CONFIG=:4096:8` env var — at a real performance cost.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (cpu + cuda + mps)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed set to %d", seed)
