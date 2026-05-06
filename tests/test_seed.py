from __future__ import annotations

import os
import random

import numpy as np
import torch

from ml_template.utils import set_seed


def test_seed_makes_torch_deterministic() -> None:
    set_seed(123)
    a = torch.randn(8)
    set_seed(123)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_seed_makes_numpy_deterministic() -> None:
    set_seed(7)
    a = np.random.rand(4)
    set_seed(7)
    b = np.random.rand(4)
    np.testing.assert_array_equal(a, b)


def test_seed_makes_python_random_deterministic() -> None:
    set_seed(9)
    a = [random.random() for _ in range(4)]
    set_seed(9)
    b = [random.random() for _ in range(4)]
    assert a == b


def test_seed_sets_pythonhashseed_env() -> None:
    set_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"
