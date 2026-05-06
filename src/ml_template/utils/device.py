"""Device resolution: cuda > mps > cpu, with explicit override support."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Accepted values for `prefer`. Kept as plain string at the type level so the
# function can be called from configs (OmegaConf structured configs don't
# fully support Literal). Unknown values raise ValueError below.
VALID_PREFS = frozenset({"auto", "cuda", "mps", "cpu"})


def resolve_device(prefer: str = "auto") -> torch.device:
    """Resolve a torch device, logging the choice at INFO.

    `auto` picks the best available accelerator. Explicit values raise
    `RuntimeError` if the requested backend is unavailable, so a typo or a
    missing CUDA install fails loudly instead of silently falling back.
    """
    if prefer not in VALID_PREFS:
        raise ValueError(
            f"Unknown device preference: {prefer!r}. Must be one of {sorted(VALID_PREFS)}."
        )
    if prefer == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info("Resolved device (auto): %s", device)
        return device

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    elif prefer == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS requested but unavailable on this build/host.")
        device = torch.device("mps")
    else:
        # prefer == "cpu" (validated above)
        device = torch.device("cpu")

    logger.info("Resolved device (explicit %s): %s", prefer, device)
    return device
