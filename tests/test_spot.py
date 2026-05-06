"""Tests for SIGTERM-survival: fsspec checkpoint upload + remote resume.

Tests use `file://` URIs because fsspec routes those through the same
abstract filesystem interface as `s3://` / `gs://`. If upload + download
work for `file://`, the cloud paths work too (auth and network errors
aside, which we don't test here).
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any

import mlflow
import pytest
from accelerate import Accelerator
from torch import nn
from torch.optim import AdamW

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    TrainerConfig,
)
from ml_template.training import train
from ml_template.training.checkpoint import (
    META_FILENAME,
    CheckpointMeta,
    save_checkpoint,
)
from ml_template.training.spot import (
    install_sigterm_handler,
    maybe_resolve_remote_resume,
    upload_checkpoint,
)


def _save_dummy_checkpoint(local_dir: Path) -> None:
    accelerator = Accelerator(cpu=True)
    model = nn.Linear(4, 2)
    optim = AdamW(model.parameters(), lr=1e-3)
    model, optim = accelerator.prepare(model, optim)
    save_checkpoint(
        accelerator,
        local_dir,
        CheckpointMeta(0, 0, "val/loss", 0.5, "min"),
    )


def _build_prepared_accelerator() -> tuple[Accelerator, nn.Module, AdamW]:
    """Accelerator with tiny prepared state — enough for save_state to work."""
    accelerator = Accelerator(cpu=True)
    model = nn.Linear(4, 2)
    optim = AdamW(model.parameters(), lr=1e-3)
    model, optim = accelerator.prepare(model, optim)
    return accelerator, model, optim


def test_upload_checkpoint_to_file_uri_round_trip(tmp_path: Path) -> None:
    """Upload to file:// then read back — verifying meta survives transit."""
    local = tmp_path / "src_ckpt"
    _save_dummy_checkpoint(local)
    remote_uri = f"file://{tmp_path / 'remote' / 'ckpt'}"

    upload_checkpoint(local, remote_uri)

    remote_path = Path(remote_uri.removeprefix("file://"))
    assert remote_path.is_dir()
    assert (remote_path / META_FILENAME).is_file()
    # Staging directory must not leak after the move.
    leftovers = list(remote_path.parent.glob("*.staging.*"))
    assert leftovers == [], leftovers


def test_upload_checkpoint_overwrites_existing(tmp_path: Path) -> None:
    """A second upload to the same URI replaces the prior contents."""
    local_v1 = tmp_path / "v1"
    local_v2 = tmp_path / "v2"
    _save_dummy_checkpoint(local_v1)
    _save_dummy_checkpoint(local_v2)
    remote_uri = f"file://{tmp_path / 'remote' / 'ckpt'}"

    upload_checkpoint(local_v1, remote_uri)
    upload_checkpoint(local_v2, remote_uri)
    # No stale staging dir.
    leftovers = list((tmp_path / "remote").glob("*.staging.*"))
    assert leftovers == [], leftovers


def test_upload_rejects_missing_local_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        upload_checkpoint(tmp_path / "no_such_dir", f"file://{tmp_path}/dst")


def test_maybe_resolve_remote_resume_passes_through_local_paths(
    tmp_path: Path,
) -> None:
    p = tmp_path / "ckpt"
    p.mkdir()
    assert maybe_resolve_remote_resume(str(p)) == p


def test_maybe_resolve_remote_resume_strips_file_scheme(tmp_path: Path) -> None:
    p = tmp_path / "ckpt"
    p.mkdir()
    resolved = maybe_resolve_remote_resume(f"file://{p}")
    assert resolved == p


def test_periodic_upload_to_checkpoint_uri(tmp_path: Path) -> None:
    """A short training run with `checkpoint_uri` set must mirror each
    periodic checkpoint to the remote URI."""
    remote_root = tmp_path / "remote"
    remote_uri = f"file://{remote_root / 'latest_ckpt'}"

    cfg = Config(
        seed=0,
        output_dir=str(tmp_path / "run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu",
            epochs=2,
            lr=5e-3,
            grad_clip_max_norm=None,
            checkpoint_every_n_epochs=1,
            keep_top_k_checkpoints=1,
            checkpoint_uri=remote_uri,
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="test_remote_upload",
        ),
    )
    train(cfg)

    # Remote path exists with metadata sidecar.
    remote_local = Path(remote_uri.removeprefix("file://"))
    assert remote_local.is_dir()
    assert (remote_local / META_FILENAME).is_file()


def test_resume_from_remote_uri_round_trip(tmp_path: Path) -> None:
    """End-to-end: train with checkpoint_uri, then resume from that
    remote URI in a fresh run."""
    remote_root = tmp_path / "remote"
    remote_uri = f"file://{remote_root / 'latest'}"

    base = Config(
        seed=0,
        output_dir=str(tmp_path / "first_run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu",
            epochs=2,
            lr=5e-3,
            grad_clip_max_norm=None,
            checkpoint_uri=remote_uri,
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="resume_remote",
        ),
    )
    train(base)

    # Resume from the remote URI in a fresh run.
    resumed = Config(
        seed=0,
        output_dir=str(tmp_path / "resumed_run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu",
            epochs=3,
            lr=5e-3,
            grad_clip_max_norm=None,
            resume_from=remote_uri,
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="resume_remote",
        ),
    )
    metrics = train(resumed)
    assert "val/loss" in metrics


# ---- SIGTERM handler tests --------------------------------------------------
# These exercise install_sigterm_handler directly — every line inside the
# handler closure was previously untested. We use os.kill on the test process;
# pytest's main thread runs the handler synchronously between bytecodes.


def test_sigterm_handler_saves_local_checkpoint_no_uri(tmp_path: Path) -> None:
    """SIGTERM with checkpoint_uri=None: save locally, exit 143."""
    accelerator, _, _ = _build_prepared_accelerator()
    output_root = tmp_path / "run"
    meta = CheckpointMeta(
        epoch=3,
        step=42,
        metric_name="val/loss",
        metric_value=0.123,
        metric_mode="min",
    )

    with (
        pytest.raises(SystemExit) as exc_info,
        install_sigterm_handler(accelerator, None, lambda: meta, output_root),
    ):
        os.kill(os.getpid(), signal.SIGTERM)

    assert exc_info.value.code == 143
    sigterm_ckpt = output_root / "sigterm_ckpt"
    assert sigterm_ckpt.is_dir()
    saved = json.loads((sigterm_ckpt / META_FILENAME).read_text())
    assert saved["epoch"] == 3
    assert saved["step"] == 42
    assert saved["metric_value"] == 0.123


def test_sigterm_handler_uploads_to_file_uri(tmp_path: Path) -> None:
    """SIGTERM with file:// URI: save locally AND upload to remote, exit 143."""
    accelerator, _, _ = _build_prepared_accelerator()
    output_root = tmp_path / "run"
    remote_uri = f"file://{tmp_path / 'remote' / 'sigterm_ckpt'}"
    meta = CheckpointMeta(0, 0, "val/loss", 0.5, "min")

    with (
        pytest.raises(SystemExit) as exc_info,
        install_sigterm_handler(accelerator, remote_uri, lambda: meta, output_root),
    ):
        os.kill(os.getpid(), signal.SIGTERM)

    assert exc_info.value.code == 143
    assert (output_root / "sigterm_ckpt" / META_FILENAME).is_file()
    remote_local = Path(remote_uri.removeprefix("file://"))
    assert (remote_local / META_FILENAME).is_file()


def test_sigterm_handler_survives_failed_upload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Bad URI: upload exception is logged but swallowed; local + exit 143 hold."""
    accelerator, _, _ = _build_prepared_accelerator()
    output_root = tmp_path / "run"
    meta = CheckpointMeta(0, 0, "val/loss", 0.5, "min")

    with (
        caplog.at_level("ERROR"),
        pytest.raises(SystemExit) as exc_info,
        install_sigterm_handler(
            accelerator, "bogus://nonexistent/path", lambda: meta, output_root
        ),
    ):
        os.kill(os.getpid(), signal.SIGTERM)

    assert exc_info.value.code == 143
    assert (output_root / "sigterm_ckpt" / META_FILENAME).is_file()
    assert "upload failed" in caplog.text.lower()


def test_sigterm_handler_restores_previous_handler(tmp_path: Path) -> None:
    """install_sigterm_handler must restore the prior signal handler in finally."""
    accelerator, _, _ = _build_prepared_accelerator()
    output_root = tmp_path / "run"

    def custom_handler(signum: int, frame: object) -> None:  # pragma: no cover
        pass

    original = signal.signal(signal.SIGTERM, custom_handler)
    try:
        with install_sigterm_handler(
            accelerator,
            None,
            lambda: CheckpointMeta(0, 0, "val/loss", 0.0, "min"),
            output_root,
        ):
            assert signal.getsignal(signal.SIGTERM) is not custom_handler
        assert signal.getsignal(signal.SIGTERM) is custom_handler
    finally:
        signal.signal(signal.SIGTERM, original)


def test_sigterm_during_training_then_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: SIGTERM mid-train(), sigterm_ckpt persists, resume works."""
    output_dir = tmp_path / "first_run"
    cfg_first = Config(
        seed=0,
        output_dir=str(output_dir),
        data=DataConfig(n_samples=32, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu",
            epochs=3,
            lr=5e-3,
            log_every_n_steps=1,
            grad_clip_max_norm=None,
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="sigterm_during_training",
        ),
    )

    # Replace mlflow.log_metric so the Nth call triggers SIGTERM. Exact placement
    # doesn't matter — what matters is the handler runs and writes sigterm_ckpt.
    original_log_metric = mlflow.log_metric
    call_count = [0]
    armed = [True]

    def killing_log_metric(*args: Any, **kwargs: Any) -> None:
        original_log_metric(*args, **kwargs)
        call_count[0] += 1
        if armed[0] and call_count[0] >= 5:
            armed[0] = False  # one shot only — the resume run must not fire it
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(mlflow, "log_metric", killing_log_metric)

    with pytest.raises(SystemExit) as exc_info:
        train(cfg_first)
    assert exc_info.value.code == 143

    sigterm_ckpt = output_dir / "sigterm_ckpt"
    assert sigterm_ckpt.is_dir()
    assert (sigterm_ckpt / META_FILENAME).is_file()

    # Resume from the sigterm_ckpt; killing_log_metric stays patched but disarmed.
    cfg_resume = Config(
        seed=0,
        output_dir=str(tmp_path / "resumed_run"),
        data=DataConfig(n_samples=32, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu",
            epochs=3,
            lr=5e-3,
            log_every_n_steps=1,
            grad_clip_max_norm=None,
            resume_from=str(sigterm_ckpt),
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="sigterm_during_training",
        ),
    )
    metrics = train(cfg_resume)
    assert "val/loss" in metrics
