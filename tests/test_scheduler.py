"""Tests for LR scheduler shape, grad accumulation, and their coupling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    TrainerConfig,
)
from ml_template.training import train
from ml_template.training.scheduler import (
    build_scheduler,
    current_lr,
    resolve_warmup_steps,
)


def _make_optim(lr: float = 1.0) -> torch.optim.Optimizer:
    """Single-param optimizer; LR shape is what we want to inspect."""
    return torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=lr)


# ---------------------------------------------------------------------------
# Pure schedule shape: warmup ramps up, cosine decays to floor.
# ---------------------------------------------------------------------------


def test_resolve_warmup_steps_uses_fraction_under_cap() -> None:
    assert resolve_warmup_steps(None, total_optim_steps=200) == 20  # 0.1 * 200


def test_resolve_warmup_steps_caps_at_default() -> None:
    assert resolve_warmup_steps(None, total_optim_steps=100_000) == 1000


def test_resolve_warmup_steps_passthrough_when_set() -> None:
    assert resolve_warmup_steps(42, total_optim_steps=10_000) == 42


def test_null_scheduler_returns_none() -> None:
    assert build_scheduler(None, _make_optim(), 100, 10, 0.1) is None


def test_linear_warmup_cosine_shape() -> None:
    optim = _make_optim(lr=1.0)
    sched = build_scheduler(
        "linear_warmup_cosine", optim, total_optim_steps=100, warmup_steps=10, min_lr_ratio=0.1
    )
    assert sched is not None

    lrs: list[float] = [current_lr(optim)]
    for _ in range(100):
        # PyTorch checks that optimizer.step() is called before
        # scheduler.step() (the first time, at least). Calling .step() with
        # no grads is a no-op but flips the internal `_opt_called` flag.
        optim.step()
        sched.step()
        lrs.append(current_lr(optim))

    # Step 0: lr is 0 (linear ramp from zero).
    assert lrs[0] == pytest.approx(0.0, abs=1e-6)
    # Step 10 (end of warmup): lr is at peak (1.0).
    assert lrs[10] == pytest.approx(1.0, abs=1e-6)
    # Within warmup the schedule is monotonically increasing.
    assert all(lrs[i] <= lrs[i + 1] for i in range(10))
    # After warmup it decays to the floor (min_lr_ratio = 0.1).
    assert lrs[100] == pytest.approx(0.1, abs=1e-3)
    # Past warmup, decreasing.
    assert all(lrs[i] >= lrs[i + 1] for i in range(10, 100))


def test_unknown_scheduler_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler("not_a_real_one", _make_optim(), 100, 10, 0.1)


def test_invalid_min_lr_ratio_raises() -> None:
    with pytest.raises(ValueError, match=r"min_lr_ratio"):
        build_scheduler("linear_warmup_cosine", _make_optim(), 100, 10, 1.5)


def test_warmup_exceeds_total_raises() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        build_scheduler("linear_warmup_cosine", _make_optim(), 100, 200, 0.1)


# ---------------------------------------------------------------------------
# Integration: scheduler steps the right number of times under grad accum.
# ---------------------------------------------------------------------------


def _make_cfg(
    tmp: Path,
    *,
    epochs: int,
    grad_accum_steps: int = 1,
    scheduler: str | None = "linear_warmup_cosine",
) -> Config:
    return Config(
        seed=0,
        output_dir=str(tmp / "run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=4),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu",
            epochs=epochs,
            lr=1e-2,
            grad_clip_max_norm=None,
            grad_accum_steps=grad_accum_steps,
            scheduler=scheduler,
            warmup_steps=2,
            min_lr_ratio=0.1,
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp / 'mlruns'}",
            experiment_name="sched",
        ),
    )


def test_scheduler_runs_end_to_end(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, epochs=2)
    metrics = train(cfg)
    assert "val/loss" in metrics


def test_grad_accum_completes_without_error(tmp_path: Path) -> None:
    """grad_accum_steps=2 must run cleanly and produce a checkpoint."""
    cfg = _make_cfg(tmp_path, epochs=2, grad_accum_steps=2)
    metrics = train(cfg)
    assert "val/loss" in metrics


def test_grad_accum_steps_validation() -> None:
    cfg = Config(
        seed=0,
        output_dir="/tmp/never_used",
        trainer=TrainerConfig(device="cpu", grad_accum_steps=0),
    )
    with pytest.raises(ValueError, match="grad_accum_steps must be >= 1"):
        train(cfg)
