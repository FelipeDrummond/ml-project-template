"""Tests for checkpoint save / load / prune and end-to-end resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from torch import nn

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    TrainerConfig,
)
from ml_template.training import (
    CheckpointMeta,
    load_checkpoint,
    prune_top_k,
    save_checkpoint,
    train,
)
from ml_template.training.checkpoint import META_FILENAME


def _make_accelerator() -> Accelerator:
    return Accelerator(cpu=True)


def _make_model_optim() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Linear(4, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, optim


def _train_one_step(
    accelerator: Accelerator, model: nn.Module, optim: torch.optim.Optimizer
) -> None:
    model.train()
    x = torch.randn(8, 4, device=accelerator.device)
    y = torch.randint(0, 2, (8,), device=accelerator.device)
    loss = nn.functional.cross_entropy(model(x), y)
    optim.zero_grad(set_to_none=True)
    accelerator.backward(loss)
    optim.step()


# ---------------------------------------------------------------------------
# Unit tests for the checkpoint module.
# ---------------------------------------------------------------------------


def test_checkpoint_meta_better_than_min_mode() -> None:
    a = CheckpointMeta(epoch=0, step=0, metric_name="val/loss", metric_value=0.5, metric_mode="min")
    b = CheckpointMeta(epoch=1, step=1, metric_name="val/loss", metric_value=0.3, metric_mode="min")
    assert b.is_better_than(a)
    assert not a.is_better_than(b)


def test_checkpoint_meta_better_than_max_mode() -> None:
    a = CheckpointMeta(epoch=0, step=0, metric_name="val/acc", metric_value=0.8, metric_mode="max")
    b = CheckpointMeta(epoch=1, step=1, metric_name="val/acc", metric_value=0.9, metric_mode="max")
    assert b.is_better_than(a)


def test_save_load_round_trip_restores_model_and_optim(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    model, optim = _make_model_optim()
    model, optim = accelerator.prepare(model, optim)
    _train_one_step(accelerator, model, optim)

    expected_state = {k: v.clone() for k, v in model.state_dict().items()}
    meta = CheckpointMeta(0, 1, "val/loss", 0.5, "min")
    ckpt = save_checkpoint(accelerator, tmp_path / "ck0", meta)
    assert ckpt.is_dir()
    assert (ckpt / META_FILENAME).is_file()

    # Mutate weights, then load — should snap back to saved state.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    loaded_meta = load_checkpoint(accelerator, ckpt)
    for k, v in expected_state.items():
        assert torch.equal(v, model.state_dict()[k]), f"Param {k} not restored"
    assert loaded_meta.epoch == 0
    assert loaded_meta.step == 1


def test_save_is_atomic(tmp_path: Path) -> None:
    """No `.staging.*` directory should remain after a successful save."""
    accelerator = _make_accelerator()
    model, optim = _make_model_optim()
    model, optim = accelerator.prepare(model, optim)
    save_checkpoint(
        accelerator,
        tmp_path / "ck0",
        CheckpointMeta(0, 0, "val/loss", 1.0, "min"),
    )
    leftovers = [p for p in tmp_path.iterdir() if ".staging." in p.name]
    assert leftovers == [], f"staging dirs leaked: {leftovers}"


def test_load_rejects_missing_meta_sidecar(tmp_path: Path) -> None:
    """A directory without checkpoint_meta.json must not be loadable."""
    bad = tmp_path / "bad_ckpt"
    bad.mkdir()
    accelerator = _make_accelerator()
    with pytest.raises(FileNotFoundError, match=META_FILENAME):
        load_checkpoint(accelerator, bad)


def test_prune_top_k_keeps_best_min_mode(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    model, optim = _make_model_optim()
    model, optim = accelerator.prepare(model, optim)

    # Save 5 checkpoints with monotonically improving val/loss.
    for i, val in enumerate([0.9, 0.7, 0.5, 0.3, 0.6]):
        save_checkpoint(
            accelerator,
            tmp_path / f"ck{i}",
            CheckpointMeta(i, i, "val/loss", val, "min"),
        )

    deleted = prune_top_k(tmp_path, keep_k=2)
    assert len(deleted) == 3

    remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    # The two best (lowest val/loss) are 0.3 and 0.5 → ck3 and ck2.
    assert remaining == ["ck2", "ck3"]


def test_prune_top_k_noop_when_under_limit(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    model, optim = _make_model_optim()
    model, optim = accelerator.prepare(model, optim)

    save_checkpoint(
        accelerator,
        tmp_path / "ck0",
        CheckpointMeta(0, 0, "val/loss", 0.5, "min"),
    )
    save_checkpoint(
        accelerator,
        tmp_path / "ck1",
        CheckpointMeta(1, 1, "val/loss", 0.3, "min"),
    )
    deleted = prune_top_k(tmp_path, keep_k=5)
    assert deleted == []


def test_prune_top_k_validates_keep_k(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        prune_top_k(tmp_path, keep_k=0)


# ---------------------------------------------------------------------------
# End-to-end resume: train → checkpoint → resume from same checkpoint →
# the resumed run produces the same final metric the un-interrupted run
# would have produced.
# ---------------------------------------------------------------------------


def _make_cfg(tmp: Path, *, epochs: int, resume_from: str | None = None) -> Config:
    return Config(
        seed=0,
        output_dir=str(tmp / "run"),
        data=DataConfig(n_samples=64, n_features=4, n_classes=2, batch_size=8),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(
            device="cpu",
            epochs=epochs,
            lr=5e-3,
            grad_clip_max_norm=None,  # avoid clipping interactions in this test
            checkpoint_every_n_epochs=1,
            keep_top_k_checkpoints=10,
            resume_from=resume_from,
        ),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp / 'mlruns'}",
            experiment_name="resume",
        ),
    )


def test_training_writes_checkpoint_per_epoch(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, epochs=3)
    train(cfg)
    ckpts = sorted((tmp_path / "run" / "checkpoints").iterdir())
    assert [p.name for p in ckpts] == ["epoch_0000", "epoch_0001", "epoch_0002"]
    for p in ckpts:
        meta = json.loads((p / META_FILENAME).read_text())
        assert meta["metric_name"] == "val/loss"
        assert meta["metric_mode"] == "min"


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    """Resuming from epoch-1 checkpoint of a 3-epoch run produces final
    metrics close to the un-interrupted 3-epoch run."""
    full = _make_cfg(tmp_path / "full", epochs=3)
    full_metrics = train(full)

    # Now: train for 2 epochs, resume from its epoch_0001 checkpoint to
    # complete the 3rd epoch.
    partial = _make_cfg(tmp_path / "partial", epochs=2)
    train(partial)
    resume_ckpt = tmp_path / "partial" / "run" / "checkpoints" / "epoch_0001"
    assert resume_ckpt.is_dir()

    resumed = _make_cfg(tmp_path / "resumed", epochs=3, resume_from=str(resume_ckpt))
    resumed_metrics = train(resumed)

    # Loss curves on this synthetic problem are deterministic enough on CPU
    # that the resumed metric should match the full-run metric to a tight
    # tolerance. Exact equality would require us to also persist the
    # dataloader iteration state (Accelerator handles that for `prepare`d
    # loaders, but only mid-epoch).
    assert abs(resumed_metrics["val/loss"] - full_metrics["val/loss"]) < 0.05, (
        f"resume drift: full={full_metrics['val/loss']:.4f} "
        f"resumed={resumed_metrics['val/loss']:.4f}"
    )
