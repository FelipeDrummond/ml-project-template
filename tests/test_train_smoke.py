"""Canonical sanity test: tiny training run must drive loss down on val.

Run after any change to the training path. CLAUDE.md mandates this.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    TrainerConfig,
)
from ml_template.training import train


def test_training_smoke_drives_val_loss_down() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            seed=0,
            output_dir=str(Path(tmp) / "run"),
            data=DataConfig(n_samples=128, n_features=8, n_classes=3, batch_size=16),
            model=ModelConfig(hidden_dim=32, n_layers=2),
            trainer=TrainerConfig(device="cpu", epochs=10, lr=5e-3),
            mlflow=MLflowConfig(
                tracking_uri=f"file:{Path(tmp) / 'mlruns'}",
                experiment_name="smoke",
            ),
        )
        metrics = train(cfg)

    assert "val/loss" in metrics
    assert "val/accuracy" in metrics
    # 3-class problem: random ≈ 0.33; we should clear that decisively.
    assert metrics["val/accuracy"] > 0.6, metrics
    assert metrics["val/loss"] < 1.0, metrics
