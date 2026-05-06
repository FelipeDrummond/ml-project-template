"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch

from ml_template.utils import set_seed


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed every test for reproducibility."""
    set_seed(0)


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")
