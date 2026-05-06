"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch

from ml_template.utils import set_seed


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed every test for reproducibility."""
    set_seed(0)


@pytest.fixture(autouse=True)
def _allow_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run on a dirty working tree by definition (you're editing
    code right now). Stub the dirty-tree gate so tests aren't blocked.
    The gate itself is exercised by tests/test_reproducibility.py.
    """
    monkeypatch.setattr(
        "ml_template.training.loop.assert_clean_or_allowed",
        lambda allow_dirty: None,
    )


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")
