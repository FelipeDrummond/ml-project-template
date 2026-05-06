"""Training loop — Accelerate-driven, MLflow-aware."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path

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


def train(cfg: Config) -> dict[str, float]:
    """Run training end-to-end. Returns final metrics for tests/callers."""
    set_seed(cfg.seed)
    enable_tf32(cfg.trainer.tf32)

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
        # Accelerator places params on the right device before optim.step;
        # `fused=True` is safe iff CUDA is the backend at that point.
        fused=accelerator.device.type == "cuda",
    )
    loss_fn = nn.CrossEntropyLoss()

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    checkpoint_root = _resolve_output_dir(cfg)
    start_epoch = 0
    best_meta: CheckpointMeta | None = None

    if cfg.trainer.resume_from is not None:
        meta = load_checkpoint(accelerator, Path(cfg.trainer.resume_from))
        start_epoch = meta.epoch + 1
        best_meta = meta

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    anomaly_ctx = (
        torch.autograd.set_detect_anomaly(True)
        if cfg.trainer.detect_anomaly
        else nullcontext()
    )

    final_metrics: dict[str, float] = {}
    with mlflow.start_run(run_name=cfg.mlflow.run_name), anomaly_ctx:
        mlflow.log_params(_flatten_params(cfg))
        global_step = 0
        for epoch in range(start_epoch, cfg.trainer.epochs):
            model.train()
            for x, y in train_loader:
                logits = model(x)
                loss = loss_fn(logits, y)
                assert_finite_loss(loss, global_step)
                optim.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                if cfg.trainer.grad_clip_max_norm is not None:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.trainer.grad_clip_max_norm
                    )
                optim.step()
                if global_step % cfg.trainer.log_every_n_steps == 0:
                    mlflow.log_metric("train/loss", loss.item(), step=global_step)
                global_step += 1

            val_loss, val_acc = _evaluate(model, val_loader, loss_fn)
            mlflow.log_metric("val/loss", val_loss, step=epoch)
            mlflow.log_metric("val/accuracy", val_acc, step=epoch)
            logger.info("epoch %d: val_loss=%.4f val_acc=%.4f", epoch, val_loss, val_acc)
            final_metrics = {"val/loss": val_loss, "val/accuracy": val_acc}

            if (epoch + 1) % cfg.trainer.checkpoint_every_n_epochs == 0:
                ckpt_meta = CheckpointMeta(
                    epoch=epoch,
                    step=global_step,
                    metric_name=TRACKED_METRIC_NAME,
                    metric_value=val_loss,
                    metric_mode=TRACKED_METRIC_MODE,
                )
                ckpt_path = checkpoint_root / f"epoch_{epoch:04d}"
                save_checkpoint(accelerator, ckpt_path, ckpt_meta)
                if best_meta is None or ckpt_meta.is_better_than(best_meta):
                    best_meta = ckpt_meta
                prune_top_k(checkpoint_root, cfg.trainer.keep_top_k_checkpoints)

    return final_metrics


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
