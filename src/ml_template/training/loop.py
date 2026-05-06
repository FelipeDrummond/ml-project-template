"""Training loop — minimal, MLflow-aware, device-explicit."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_template.config_schemas import Config
from ml_template.data import build_dataloaders
from ml_template.models import MLP
from ml_template.utils import (
    assert_finite_loss,
    enable_tf32,
    resolve_device,
    set_seed,
    use_fused_adamw,
)

logger = logging.getLogger(__name__)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """Return (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.inference_mode():
        for x, y in loader:
            x_dev = x.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True)
            logits = model(x_dev)
            loss = loss_fn(logits, y_dev)
            total_loss += loss.item() * x_dev.shape[0]
            total_correct += int((logits.argmax(dim=-1) == y_dev).sum().item())
            total_seen += x_dev.shape[0]
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def train(cfg: Config) -> dict[str, float]:
    """Run training end-to-end. Returns final metrics for tests/callers."""
    set_seed(cfg.seed)
    enable_tf32(cfg.trainer.tf32)
    device = resolve_device(cfg.trainer.device)

    train_loader, val_loader = build_dataloaders(cfg.data, seed=cfg.seed)

    model = MLP(
        in_features=cfg.data.n_features,
        out_features=cfg.data.n_classes,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
        fused=use_fused_adamw(),
    )
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Anomaly detection is dev-only — very slow but pinpoints the op that
    # produced a NaN/Inf. Use it once to find the bug, then disable.
    anomaly_ctx = (
        torch.autograd.set_detect_anomaly(True)
        if cfg.trainer.detect_anomaly
        else nullcontext()
    )

    final_metrics: dict[str, float] = {}
    with mlflow.start_run(run_name=cfg.mlflow.run_name), anomaly_ctx:
        mlflow.log_params(_flatten_params(cfg))
        global_step = 0
        for epoch in range(cfg.trainer.epochs):
            model.train()
            for x, y in train_loader:
                x_dev = x.to(device, non_blocking=True)
                y_dev = y.to(device, non_blocking=True)
                logits = model(x_dev)
                loss = loss_fn(logits, y_dev)
                assert_finite_loss(loss, global_step)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.trainer.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.trainer.grad_clip_max_norm
                    )
                optim.step()
                if global_step % cfg.trainer.log_every_n_steps == 0:
                    mlflow.log_metric("train/loss", loss.item(), step=global_step)
                global_step += 1

            val_loss, val_acc = _evaluate(model, val_loader, device, loss_fn)
            mlflow.log_metric("val/loss", val_loss, step=epoch)
            mlflow.log_metric("val/accuracy", val_acc, step=epoch)
            logger.info("epoch %d: val_loss=%.4f val_acc=%.4f", epoch, val_loss, val_acc)
            final_metrics = {"val/loss": val_loss, "val/accuracy": val_acc}

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
