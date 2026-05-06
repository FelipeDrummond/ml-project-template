"""Print which torch backends are available on this host."""

from __future__ import annotations

import torch

print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    cuda_build = getattr(torch.version, "cuda", "unknown")  # pyright: ignore[reportAttributeAccessIssue]
    print(f"  CUDA version (build): {cuda_build}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built:     {torch.backends.mps.is_built()}")
