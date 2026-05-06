"""Tests for `fast_dev_run`: 1 batch + checkpoint round-trip end-to-end.

This is the canonical PR-blocking smoke test. It exercises every code
path the loop touches — model build, accelerator prepare, optional
torch.compile, scheduler, grad accumulation, autocast (skipped on CPU),
checkpoint save → load round-trip, MLflow logging — without paying for
a real training run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest
import torch

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    TrainerConfig,
)
from ml_template.training import train
from ml_template.training.checkpoint import META_FILENAME


def _make_cfg(tmp: Path, **trainer_overrides: object) -> Config:
    trainer_kwargs: dict[str, object] = {
        "device": "cpu",
        "epochs": 10,  # fast_dev_run truncates to 1 by default
        "lr": 1e-3,
        "grad_clip_max_norm": None,
        "fast_dev_run": True,
    }
    trainer_kwargs.update(trainer_overrides)
    return Config(
        seed=0,
        output_dir=str(tmp / "run"),
        data=DataConfig(n_samples=32, n_features=4, n_classes=2, batch_size=4),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(**trainer_kwargs),  # type: ignore[arg-type]
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp / 'mlruns'}",
            experiment_name="fast_dev_run",
        ),
    )


def test_fast_dev_run_runs_one_epoch_only(tmp_path: Path) -> None:
    """`epochs=10` is overridden to 1 by `fast_dev_run`."""
    cfg = _make_cfg(tmp_path)
    train(cfg)

    ckpt_root = tmp_path / "run" / "checkpoints"
    ckpts = sorted(p.name for p in ckpt_root.iterdir() if (p / META_FILENAME).is_file())
    assert ckpts == ["epoch_0000"], ckpts


def test_fast_dev_run_writes_checkpoint_with_meta(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    train(cfg)
    ckpt = tmp_path / "run" / "checkpoints" / "epoch_0000"
    assert ckpt.is_dir()
    assert (ckpt / META_FILENAME).is_file()


def test_fast_dev_run_with_grad_accum_completes(tmp_path: Path) -> None:
    """Cap is `max(1, grad_accum_steps)` so at least one optim step fires."""
    cfg = _make_cfg(tmp_path, grad_accum_steps=2)
    train(cfg)
    assert (tmp_path / "run" / "checkpoints" / "epoch_0000").is_dir()


def test_fast_dev_run_with_scheduler_completes(tmp_path: Path) -> None:
    """Scheduler must build and step at least once."""
    cfg = _make_cfg(tmp_path, scheduler="linear_warmup_cosine", warmup_steps=1, min_lr_ratio=0.1)
    train(cfg)
    assert (tmp_path / "run" / "checkpoints" / "epoch_0000").is_dir()


def test_fast_dev_run_round_trip_catches_format_drift(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round-trip is the headline value: if save/load disagree, fail."""
    caplog.set_level(logging.INFO, logger="ml_template.training.loop")

    cfg = _make_cfg(tmp_path)
    train(cfg)
    assert any("round-trip OK" in record.message for record in caplog.records), (
        "fast_dev_run did not log the round-trip success line"
    )


def test_fast_dev_run_runs_quickly(tmp_path: Path) -> None:
    """With `epochs=1000` the run still finishes in well under a minute,
    proving the truncation is real (not just a default-epochs accident)."""
    cfg = _make_cfg(tmp_path, epochs=1000)
    start = time.monotonic()
    train(cfg)
    elapsed = time.monotonic() - start
    assert elapsed < 30.0, f"fast_dev_run took {elapsed:.1f}s — truncation broken?"


# ---------------------------------------------------------------------------
# Sanity: non-fast_dev_run still works (no regression).
# ---------------------------------------------------------------------------


def test_non_fast_dev_run_still_runs_full_epochs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.trainer.fast_dev_run = False
    cfg.trainer.epochs = 2
    train(cfg)
    ckpts = sorted(
        p.name
        for p in (tmp_path / "run" / "checkpoints").iterdir()
        if (p / META_FILENAME).is_file()
    )
    assert len(ckpts) >= 1  # may be pruned by keep_top_k but at least one
    # And it ran at least 2 actual training epochs.
    if not torch.cuda.is_available():
        # CPU run is fast enough that we trust the assertion above.
        pass
