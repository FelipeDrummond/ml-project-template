"""Training loop — Accelerate-driven, MLflow-aware."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import cast

import mlflow
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from ml_template.config_schemas import Config
from ml_template.data import build_dataloaders
from ml_template.models import MLP
from ml_template.training.checkpoint import (
    CheckpointMeta,
    load_checkpoint,
    prune_top_k,
    save_checkpoint,
)
from ml_template.training.spot import (
    install_sigterm_handler,
    maybe_resolve_remote_resume,
    upload_checkpoint,
)
from ml_template.utils import (
    assert_finite_loss,
    enable_tf32,
    set_seed,
)

logger = logging.getLogger(__name__)

# We optimize val/loss → smaller is better. If you switch to a metric where
# bigger is better (e.g. accuracy), change this in one place.
TRACKED_METRIC_NAME = "val/loss"
TRACKED_METRIC_MODE = "min"


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """Return (mean_loss, accuracy). Tensors are already on-device (Accelerator)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.shape[0]
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_seen += x.shape[0]
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def _train_one_epoch(
    cfg: Config,
    accelerator: Accelerator,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    loss_fn: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    starting_step: int,
) -> int:
    """Run one training epoch. Returns the new global_step counter."""
    model.train()
    step = starting_step
    for x, y in loader:
        logits = model(x)
        loss = loss_fn(logits, y)
        assert_finite_loss(loss, step)
        optim.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        if cfg.trainer.grad_clip_max_norm is not None:
            accelerator.clip_grad_norm_(
                model.parameters(), cfg.trainer.grad_clip_max_norm
            )
        optim.step()
        if step % cfg.trainer.log_every_n_steps == 0:
            mlflow.log_metric("train/loss", loss.item(), step=step)
        step += 1
    return step


def _build_accelerator(cfg: Config) -> Accelerator:
    """Construct the Accelerator with config-driven precision and CPU override.

    `device == "cpu"` forces `cpu=True`; everything else lets Accelerate
    auto-detect (CUDA > MPS > CPU). MPS does not support bf16/fp16 autocast
    cleanly, so we silently downgrade to "no" precision when MPS is active.
    """
    cpu_only = cfg.trainer.device == "cpu"
    precision = cfg.trainer.precision
    if not cpu_only and not torch.cuda.is_available() and precision != "no":
        logger.warning(
            "Accelerator: requested mixed_precision=%r but no CUDA detected; "
            "downgrading to 'no' (MPS/CPU don't autocast cleanly).",
            precision,
        )
        precision = "no"
    return Accelerator(mixed_precision=precision, cpu=cpu_only)


def _resolve_output_dir(cfg: Config) -> Path:
    """Where checkpoints go. Tests pass `cfg.output_dir`; CLI fills it
    from Hydra's runtime output dir."""
    if cfg.output_dir is None:
        # Fallback: a sibling of the MLflow tracking dir. Useful in ad-hoc
        # invocations from a REPL where Hydra didn't run.
        return Path("outputs/run") / "checkpoints"
    return Path(cfg.output_dir) / "checkpoints"


@dataclass
class _TrainState:
    accelerator: Accelerator
    model: nn.Module
    optim: torch.optim.Optimizer
    loss_fn: nn.Module
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    checkpoint_root: Path
    start_epoch: int
    best_meta: CheckpointMeta | None


def _setup(cfg: Config) -> _TrainState:
    """Build everything the training loop needs: accelerator, model+optim
    (prepared and optionally compiled), dataloaders, and resume state."""
    accelerator = _build_accelerator(cfg)
    logger.info("Accelerator device: %s, precision: %s",
                accelerator.device, accelerator.mixed_precision)

    train_loader, val_loader = build_dataloaders(cfg.data, seed=cfg.seed)
    model = MLP(
        in_features=cfg.data.n_features,
        out_features=cfg.data.n_classes,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
        # `fused=True` is safe iff CUDA is the backend at optim.step.
        fused=accelerator.device.type == "cuda",
    )
    loss_fn = nn.CrossEntropyLoss()

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    # torch.compile is CUDA-only. Silently skip with a warning elsewhere
    # to avoid compile overhead with no payoff.
    if cfg.trainer.compile_mode is not None:
        if accelerator.device.type == "cuda":
            logger.info("Compiling model with torch.compile(mode=%r)",
                        cfg.trainer.compile_mode)
            # torch.compile returns OptimizedModule (an nn.Module subclass)
            # but pyright stubs type it as Callable; cast back so downstream
            # code remains nn.Module-typed.
            model = cast(nn.Module, torch.compile(model, mode=cfg.trainer.compile_mode))
        else:
            logger.warning(
                "compile_mode=%r requested but device is %s; skipping torch.compile.",
                cfg.trainer.compile_mode, accelerator.device.type,
            )

    start_epoch = 0
    best_meta: CheckpointMeta | None = None
    if cfg.trainer.resume_from is not None:
        # `resume_from` may be a local path or a remote fsspec URI
        # (s3://, gs://...). Remote URIs are mirrored locally first.
        resume_path = maybe_resolve_remote_resume(cfg.trainer.resume_from)
        meta = load_checkpoint(accelerator, resume_path)
        start_epoch = meta.epoch + 1
        best_meta = meta

    return _TrainState(
        accelerator=accelerator,
        model=model,
        optim=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_root=_resolve_output_dir(cfg),
        start_epoch=start_epoch,
        best_meta=best_meta,
    )


def train(cfg: Config) -> dict[str, float]:
    """Run training end-to-end. Returns final metrics for tests/callers."""
    set_seed(cfg.seed)
    enable_tf32(cfg.trainer.tf32)
    s = _setup(cfg)
    accelerator = s.accelerator  # used in many places below; alias once
    best_meta = s.best_meta

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    anomaly_ctx = (
        torch.autograd.set_detect_anomaly(True)
        if cfg.trainer.detect_anomaly
        else nullcontext()
    )

    final_metrics: dict[str, float] = {}
    epochs_since_improvement = 0
    train_start = time.monotonic()

    # Mutable container shared with the SIGTERM handler so it can take a
    # checkpoint of the *current* state when the spot box is reclaimed.
    last_state: dict[str, object] = {
        "epoch": s.start_epoch,
        "step": 0,
        "val_loss": float("inf"),
    }
    sigterm_ctx = _build_sigterm_ctx(
        cfg, accelerator, s.checkpoint_root.parent, last_state
    )

    with mlflow.start_run(run_name=cfg.mlflow.run_name), anomaly_ctx, sigterm_ctx:
        mlflow.log_params(_flatten_params(cfg))
        global_step = 0
        for epoch in range(s.start_epoch, cfg.trainer.epochs):
            global_step = _train_one_epoch(
                cfg, accelerator, s.model, s.optim, s.loss_fn,
                s.train_loader, global_step,
            )
            val_loss, val_acc = _evaluate(s.model, s.val_loader, s.loss_fn)
            mlflow.log_metric("val/loss", val_loss, step=epoch)
            mlflow.log_metric("val/accuracy", val_acc, step=epoch)
            logger.info("epoch %d: val_loss=%.4f val_acc=%.4f", epoch, val_loss, val_acc)
            final_metrics = {"val/loss": val_loss, "val/accuracy": val_acc}

            ckpt_meta = CheckpointMeta(
                epoch=epoch,
                step=global_step,
                metric_name=TRACKED_METRIC_NAME,
                metric_value=val_loss,
                metric_mode=TRACKED_METRIC_MODE,
            )
            if best_meta is None or ckpt_meta.is_better_than(best_meta):
                best_meta = ckpt_meta
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            last_state["epoch"] = epoch
            last_state["step"] = global_step
            last_state["val_loss"] = val_loss

            if (epoch + 1) % cfg.trainer.checkpoint_every_n_epochs == 0:
                _save_and_maybe_upload(cfg, accelerator, s.checkpoint_root, ckpt_meta)

            # Cost controls run after the checkpoint, so an early exit
            # always leaves a fresh on-disk checkpoint to resume from.
            stop_reason = _should_stop(
                cfg, epoch, epochs_since_improvement, train_start
            )
            if stop_reason is not None:
                final_metrics[stop_reason[0]] = stop_reason[1]
                break

    return final_metrics


def _save_and_maybe_upload(
    cfg: Config,
    accelerator: Accelerator,
    checkpoint_root: Path,
    meta: CheckpointMeta,
) -> None:
    """Save the periodic checkpoint, prune to top-K, and (if configured)
    mirror the latest one to the spot-survival URI."""
    ckpt_path = checkpoint_root / f"epoch_{meta.epoch:04d}"
    save_checkpoint(accelerator, ckpt_path, meta)
    prune_top_k(checkpoint_root, cfg.trainer.keep_top_k_checkpoints)
    if cfg.trainer.checkpoint_uri is not None:
        # Best-effort: upload failures are logged but don't kill the run.
        try:
            upload_checkpoint(ckpt_path, cfg.trainer.checkpoint_uri)
        except Exception:
            logger.exception("Periodic checkpoint upload failed.")


def _build_sigterm_ctx(
    cfg: Config,
    accelerator: Accelerator,
    output_root: Path,
    last_state: dict[str, object],
):
    """Return a context manager that installs the SIGTERM handler.

    The handler reads `last_state` (mutated by the training loop each
    epoch) so the checkpoint it writes reflects the most recent step.
    """

    def take_sigterm_checkpoint() -> CheckpointMeta:
        return CheckpointMeta(
            epoch=int(last_state["epoch"]),  # type: ignore[arg-type]
            step=int(last_state["step"]),  # type: ignore[arg-type]
            metric_name=TRACKED_METRIC_NAME,
            metric_value=float(last_state["val_loss"]),  # type: ignore[arg-type]
            metric_mode=TRACKED_METRIC_MODE,
        )

    return install_sigterm_handler(
        accelerator, cfg.trainer.checkpoint_uri, take_sigterm_checkpoint, output_root
    )


def _should_stop(
    cfg: Config,
    epoch: int,
    epochs_since_improvement: int,
    train_start: float,
) -> tuple[str, float] | None:
    """Decide whether to exit the training loop after this epoch.

    Returns a (metric_key, metric_value) pair to record in `final_metrics`,
    or `None` to continue training.
    """
    if (
        cfg.trainer.early_stop_patience is not None
        and epochs_since_improvement >= cfg.trainer.early_stop_patience
    ):
        logger.info(
            "Early stopping at epoch %d: %s has not improved for %d epochs.",
            epoch, TRACKED_METRIC_NAME, epochs_since_improvement,
        )
        return ("stopped_at_epoch", float(epoch))

    elapsed = time.monotonic() - train_start
    if (
        cfg.trainer.max_wall_seconds is not None
        and elapsed >= cfg.trainer.max_wall_seconds
    ):
        logger.info(
            "Wall-clock budget exhausted at epoch %d (%.1fs >= %ds). Exiting cleanly.",
            epoch, elapsed, cfg.trainer.max_wall_seconds,
        )
        return ("wall_seconds", elapsed)

    return None


def _flatten_params(cfg: Config) -> dict[str, str]:
    """Flatten nested dataclass to dotted keys for MLflow params."""
    if not is_dataclass(cfg):
        raise TypeError(f"Expected a dataclass, got {type(cfg).__name__}")

    flat: dict[str, str] = {}

    def walk(prefix: str, obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            flat[prefix] = str(obj)

    walk("", asdict(cfg))
    return flat
