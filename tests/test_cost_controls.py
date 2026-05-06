"""Tests for early stopping and max-wall-clock budget."""

from __future__ import annotations

from pathlib import Path

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    TrainerConfig,
)
from ml_template.training import train
from ml_template.training.checkpoint import META_FILENAME


def _make_cfg(
    tmp: Path,
    *,
    epochs: int,
    early_stop_patience: int | None = None,
    max_wall_seconds: int | None = None,
    lr: float = 5e-3,
) -> Config:
    return Config(
        seed=0,
        output_dir=str(tmp / "run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu",
            epochs=epochs,
            lr=lr,
            grad_clip_max_norm=None,
            checkpoint_every_n_epochs=1,
            keep_top_k_checkpoints=10,
            early_stop_patience=early_stop_patience,
            max_wall_seconds=max_wall_seconds,
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp / 'mlruns'}",
            experiment_name="cost_controls",
        ),
    )


def _checkpoint_count(tmp: Path) -> int:
    ckpt_root = tmp / "run" / "checkpoints"
    if not ckpt_root.is_dir():
        return 0
    return sum(1 for p in ckpt_root.iterdir() if (p / META_FILENAME).is_file())


def test_no_cost_controls_runs_all_epochs(tmp_path: Path) -> None:
    """Baseline: with both controls disabled, training runs the full
    requested number of epochs."""
    cfg = _make_cfg(tmp_path, epochs=4)
    train(cfg)
    assert _checkpoint_count(tmp_path) == 4


def test_early_stop_triggers_when_metric_plateaus(tmp_path: Path) -> None:
    """With a tiny lr and patience=1, val/loss stalls quickly and
    training exits before the configured epoch count."""
    # Very low LR so val/loss barely changes — patience=1 will fire.
    cfg = _make_cfg(tmp_path, epochs=20, early_stop_patience=1, lr=1e-8)
    train(cfg)
    # Should NOT have run all 20 epochs.
    assert _checkpoint_count(tmp_path) < 20


def test_early_stop_does_not_trigger_on_steady_improvement(tmp_path: Path) -> None:
    """With healthy lr, val/loss keeps improving — patience should not
    fire, training runs to completion."""
    cfg = _make_cfg(tmp_path, epochs=3, early_stop_patience=2, lr=5e-3)
    train(cfg)
    assert _checkpoint_count(tmp_path) == 3


def test_max_wall_seconds_exits_cleanly(tmp_path: Path) -> None:
    """With max_wall_seconds=0, training exits after the first epoch
    (the check happens after epoch end), and a checkpoint is on disk
    so the run is resumable."""
    cfg = _make_cfg(tmp_path, epochs=10, max_wall_seconds=0)
    metrics = train(cfg)
    assert "wall_seconds" in metrics
    # At least one checkpoint must exist — the spot-friendly invariant.
    assert _checkpoint_count(tmp_path) >= 1
    # And we did NOT run all 10 epochs.
    assert _checkpoint_count(tmp_path) < 10
