"""Tests for Tier-1 perf knobs.

Each test pins one behavior so a regression in the perf path produces a
clear, isolated failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pytest
import torch

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    TrainerConfig,
)
from ml_template.data import build_dataloaders
from ml_template.training import train
from ml_template.utils.perf import (
    assert_finite_loss,
    enable_tf32,
    gpu_memory_snapshot,
    reset_peak_memory_stats,
    use_fused_adamw,
)


def test_enable_tf32_sets_high_precision() -> None:
    enable_tf32(True)
    assert torch.get_float32_matmul_precision() == "high"


def test_enable_tf32_false_sets_highest_precision() -> None:
    enable_tf32(False)
    assert torch.get_float32_matmul_precision() == "highest"
    enable_tf32(True)  # restore for other tests


def test_use_fused_adamw_matches_cuda_availability() -> None:
    assert use_fused_adamw() == torch.cuda.is_available()


def test_assert_finite_loss_passes_on_finite() -> None:
    assert_finite_loss(torch.tensor(0.123), step=0)


def test_assert_finite_loss_raises_on_nan() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        assert_finite_loss(torch.tensor(float("nan")), step=42)


def test_assert_finite_loss_raises_on_inf() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        assert_finite_loss(torch.tensor(float("inf")), step=42)


def test_gpu_memory_snapshot_empty_off_cuda() -> None:
    """Empty dict on non-CUDA hosts (the test happy-path on Mac/CI)."""
    if torch.cuda.is_available():
        pytest.skip("This test pins the non-CUDA fallback.")
    assert gpu_memory_snapshot() == {}


def test_reset_peak_memory_stats_no_op_off_cuda() -> None:
    """Must not raise on non-CUDA hosts."""
    if torch.cuda.is_available():
        pytest.skip("This test pins the non-CUDA fallback.")
    reset_peak_memory_stats()  # would crash if unguarded


def test_perf_metrics_logged_to_mlflow(tmp_path: Path) -> None:
    """Run a short training and assert the perf metrics make it into the
    MLflow file store."""
    cfg = Config(
        seed=0,
        output_dir=str(tmp_path / "run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu", epochs=1, lr=1e-3, grad_clip_max_norm=None
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="perf_metrics",
        ),
    )
    train(cfg)

    client = mlflow.MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name("perf_metrics")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    metrics = runs[0].data.metrics
    assert "perf/samples_per_sec" in metrics
    assert metrics["perf/samples_per_sec"] > 0
    assert "perf/dataloader_wait_pct" in metrics
    assert 0.0 <= metrics["perf/dataloader_wait_pct"] <= 100.0
    assert "perf/epoch_seconds" in metrics
    assert metrics["perf/epoch_seconds"] > 0


def test_dataloader_workers_gated_off_without_cuda() -> None:
    """On a CUDA-less host, num_workers / pin_memory must be silently
    forced to safe values regardless of what the user requests."""
    if torch.cuda.is_available():
        pytest.skip("This test pins macOS / CPU-only behavior.")

    cfg = DataConfig(
        n_samples=64,
        n_features=4,
        n_classes=2,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader, _ = build_dataloaders(cfg, seed=0)
    assert train_loader.num_workers == 0
    assert train_loader.pin_memory is False


def test_compile_mode_silently_skipped_on_non_cuda(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`compile_mode` is CUDA-only. On Mac/CPU it must log a warning and
    skip rather than crashing or silently slowing training."""
    if torch.cuda.is_available():
        pytest.skip("This test pins the non-CUDA fallback behavior.")

    caplog.set_level(logging.WARNING, logger="ml_template.training.loop")
    cfg = Config(
        seed=0,
        output_dir=str(tmp_path / "run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu", epochs=1, lr=1e-3, compile_mode="default"
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="test_compile_skip",
        ),
    )
    train(cfg)
    assert any(
        "skipping torch.compile" in record.message for record in caplog.records
    ), [r.message for r in caplog.records]


def test_grad_clip_can_be_disabled(tmp_path: Path) -> None:
    """`grad_clip_max_norm=null` must run without clipping (no crash)."""
    cfg = Config(
        seed=0,
        output_dir=str(tmp_path / "run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu", epochs=1, lr=1e-3, grad_clip_max_norm=None
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="test_clip_off",
        ),
    )
    metrics = train(cfg)
    assert "val/loss" in metrics
