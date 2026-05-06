from __future__ import annotations

import pytest
import torch

from ml_template.utils.device import resolve_device


def test_cpu_always_works() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_returns_a_real_device() -> None:
    device = resolve_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}


def test_unknown_pref_raises() -> None:
    with pytest.raises(ValueError, match="Unknown device preference"):
        resolve_device("tpu")


def test_explicit_cuda_when_unavailable_raises() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available — cannot test the failure path here.")
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")
