"""Training loop — Accelerate-driven, MLflow-aware."""

from __future__ import annotations

import logging
import math
import sys
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
from ml_template.training.reproducibility import (
    assert_clean_or_allowed,
    env_summary,
    freeze_packages,
    git_diff,
    git_state,
)
from ml_template.training.scheduler import (
    build_scheduler,
    current_lr,
    resolve_warmup_steps,
)
from ml_template.training.spot import (
    install_sigterm_handler,
    maybe_resolve_remote_resume,
    upload_checkpoint,
)
from ml_template.utils import (
    enable_tf32,
    gpu_memory_snapshot,
    reset_peak_memory_stats,
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
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Return (mean_loss, accuracy). `max_batches` truncates for fast_dev_run."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    starting_step: int,
    max_batches: int | None = None,
) -> tuple[int, int, float, float]:
    """Run one training epoch. Returns (new_step, samples_seen, data_wait_s, compute_s).

    `step` counts micro-batches. With grad_accum_steps > 1, the optimizer
    steps once per N micro-batches; `accelerator.accumulate(model)` gates
    the actual backward sync + optim/scheduler step on the boundary.
    `max_batches` truncates the loop for fast_dev_run.

    Timing semantics: `data_wait_s` is wall time from the end of the
    previous compute block to the start of the next iteration's compute —
    i.e. how long the main thread blocked on the dataloader. High % here
    means more workers / pin_memory / smaller per-sample preprocessing.
    """
    model.train()
    step = starting_step
    samples_seen = 0
    data_wait_s = 0.0
    compute_s = 0.0
    last_t = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        now = time.perf_counter()
        data_wait_s += now - last_t
        if max_batches is not None and i >= max_batches:
            break
        with accelerator.accumulate(model):
            logits = model(x)
            loss = loss_fn(logits, y)
            accelerator.backward(loss)
            if accelerator.sync_gradients and cfg.trainer.grad_clip_max_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip_max_norm)
            optim.step()
            if scheduler is not None:
                scheduler.step()
            optim.zero_grad(set_to_none=True)
        last_t = time.perf_counter()
        compute_s += last_t - now
        samples_seen += x.shape[0]
        if step % cfg.trainer.log_every_n_steps == 0:
            # Single .item() pays the only CPU↔GPU sync we accept in the
            # hot loop; piggyback the NaN/Inf guard on it. Worst-case
            # detection latency is log_every_n_steps; cheaper than a
            # per-step isfinite sync.
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"Loss is non-finite at step {step}: {loss_value!r}. "
                    f"Set trainer.detect_anomaly=true to locate the offending op."
                )
            mlflow.log_metric("train/loss", loss_value, step=step)
            mlflow.log_metric("train/lr", current_lr(optim), step=step)
        step += 1
    return step, samples_seen, data_wait_s, compute_s


def _build_accelerator(cfg: Config) -> Accelerator:
    """Construct the Accelerator with config-driven precision and CPU override.

    `device == "cpu"` forces `cpu=True`; everything else lets Accelerate
    auto-detect (CUDA > MPS > CPU). MPS does not support bf16/fp16 autocast
    cleanly, so we silently downgrade to "no" precision when MPS is active.

    `gradient_accumulation_steps` enables `accelerator.accumulate(model)` —
    correctly suppresses DDP grad-sync on micro-steps even on single-GPU.
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
    if cfg.trainer.grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {cfg.trainer.grad_accum_steps}")
    return Accelerator(
        mixed_precision=precision,
        cpu=cpu_only,
        gradient_accumulation_steps=cfg.trainer.grad_accum_steps,
    )


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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    loss_fn: nn.Module
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    checkpoint_root: Path
    start_epoch: int
    best_meta: CheckpointMeta | None


def _setup(cfg: Config) -> _TrainState:
    """Build everything the training loop needs: accelerator, model+optim
    (prepared and optionally compiled), dataloaders, and resume state.

    Also runs the reproducibility gate first (refuses to spend money on
    a run we can't reconstruct) and the seed/TF32 init.
    """
    # Reproducibility gate first — `run.allow_dirty=true` is the escape hatch.
    assert_clean_or_allowed(cfg.run.allow_dirty)
    set_seed(cfg.seed)
    enable_tf32()
    accelerator = _build_accelerator(cfg)
    logger.info(
        "Accelerator device: %s, precision: %s", accelerator.device, accelerator.mixed_precision
    )

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

    # Scheduler total step count is in *optimizer* steps (post-accumulation).
    total_optim_steps = (len(train_loader) // cfg.trainer.grad_accum_steps) * cfg.trainer.epochs
    warmup_steps = resolve_warmup_steps(cfg.trainer.warmup_steps, total_optim_steps)
    scheduler = build_scheduler(
        cfg.trainer.scheduler,
        optim,
        total_optim_steps,
        warmup_steps,
        cfg.trainer.min_lr_ratio,
    )

    if scheduler is not None:
        model, optim, scheduler, train_loader, val_loader = accelerator.prepare(
            model, optim, scheduler, train_loader, val_loader
        )
    else:
        model, optim, train_loader, val_loader = accelerator.prepare(
            model, optim, train_loader, val_loader
        )

    # torch.compile is CUDA-only. Silently skip with a warning elsewhere
    # to avoid compile overhead with no payoff.
    if cfg.trainer.compile_mode is not None:
        if accelerator.device.type == "cuda":
            logger.info("Compiling model with torch.compile(mode=%r)", cfg.trainer.compile_mode)
            # torch.compile returns OptimizedModule (an nn.Module subclass)
            # but pyright stubs type it as Callable; cast back so downstream
            # code remains nn.Module-typed.
            model = cast(nn.Module, torch.compile(model, mode=cfg.trainer.compile_mode))
        else:
            logger.warning(
                "compile_mode=%r requested but device is %s; skipping torch.compile.",
                cfg.trainer.compile_mode,
                accelerator.device.type,
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
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_root=_resolve_output_dir(cfg),
        start_epoch=start_epoch,
        best_meta=best_meta,
    )


def train(cfg: Config) -> dict[str, float]:
    """Run training end-to-end. Returns final metrics for tests/callers."""
    s = _setup(cfg)
    accelerator = s.accelerator  # used in many places below; alias once
    best_meta = s.best_meta

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    anomaly_ctx = (
        torch.autograd.set_detect_anomaly(True) if cfg.trainer.detect_anomaly else nullcontext()
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
    sigterm_ctx = _build_sigterm_ctx(cfg, accelerator, s.checkpoint_root.parent, last_state)

    fdr = cfg.trainer.fast_dev_run
    epochs = 1 if fdr else cfg.trainer.epochs
    train_cap = max(1, cfg.trainer.grad_accum_steps) if fdr else None
    val_cap = 1 if fdr else None
    if fdr:
        logger.warning(
            "fast_dev_run: 1 epoch, %d train batch(es), %d val batch(es), "
            "forced checkpoint + load round-trip.",
            train_cap,
            val_cap,
        )

    with mlflow.start_run(run_name=cfg.mlflow.run_name), anomaly_ctx, sigterm_ctx:
        _log_reproducibility_envelope(cfg)
        global_step = 0
        for epoch in range(s.start_epoch, epochs):
            reset_peak_memory_stats()
            global_step, samples_seen, data_wait_s, compute_s = _train_one_epoch(
                cfg,
                accelerator,
                s.model,
                s.optim,
                s.scheduler,
                s.loss_fn,
                s.train_loader,
                global_step,
                max_batches=train_cap,
            )
            val_loss, val_acc = _evaluate(s.model, s.val_loader, s.loss_fn, max_batches=val_cap)
            mlflow.log_metric("val/loss", val_loss, step=epoch)
            mlflow.log_metric("val/accuracy", val_acc, step=epoch)
            _log_epoch_perf(epoch, samples_seen, data_wait_s, compute_s)
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

            # fast_dev_run forces a save+load round-trip regardless of
            # checkpoint_every_n_epochs; that's the bug it most often catches.
            if fdr or (epoch + 1) % cfg.trainer.checkpoint_every_n_epochs == 0:
                ckpt_path = _save_and_maybe_upload(cfg, accelerator, s.checkpoint_root, ckpt_meta)
                if fdr:
                    load_checkpoint(accelerator, ckpt_path)
                    logger.info("fast_dev_run: checkpoint round-trip OK.")

            # Cost controls run after the checkpoint, so an early exit
            # always leaves a fresh on-disk checkpoint to resume from.
            stop_reason = _should_stop(cfg, epoch, epochs_since_improvement, train_start)
            if stop_reason is not None:
                final_metrics[stop_reason[0]] = stop_reason[1]
                break

    return final_metrics


def _log_reproducibility_envelope(cfg: Config) -> None:
    """Log git state + env + cli + diff + freeze to the active MLflow run.

    Splits naturally into:
    - small string params (sha, branch, dirty flag, env summary, argv)
    - artifacts (full diff if dirty, package freeze, resolved config)
    """
    params: dict[str, str] = {}
    params.update(git_state())
    params.update(env_summary())
    params["run/argv"] = " ".join(sys.argv)
    # MLflow caps params at 500 chars; truncate defensively.
    truncated = {k: (v[:500] if isinstance(v, str) else v) for k, v in params.items()}
    mlflow.log_params(truncated)

    # Artifacts: diff (only when dirty), pip freeze, resolved config.
    if params.get("git/dirty") == "true":
        diff = git_diff()
        if diff:
            mlflow.log_text(diff, "git_diff.patch")
    freeze = freeze_packages()
    if freeze:
        mlflow.log_text(freeze, "packages.txt")
    mlflow.log_text(_render_resolved_config(cfg), "resolved_config.txt")


def _render_resolved_config(cfg: Config) -> str:
    """Flat dotted-key dump of the resolved (post-overrides) config."""
    return "\n".join(f"{k}={v}" for k, v in sorted(_flatten_params(cfg).items()))


def _log_epoch_perf(epoch: int, samples_seen: int, data_wait_s: float, compute_s: float) -> None:
    """Log throughput, dataloader-wait %, and GPU memory/util to MLflow.

    `data_wait_pct` is the headline cost-of-IO number — over ~20% means
    the dataloader is the bottleneck (more workers / pin_memory / smaller
    per-sample preprocessing).
    """
    total = data_wait_s + compute_s
    if total > 0:
        mlflow.log_metric("perf/samples_per_sec", samples_seen / total, step=epoch)
        mlflow.log_metric("perf/dataloader_wait_pct", 100.0 * data_wait_s / total, step=epoch)
        mlflow.log_metric("perf/epoch_seconds", total, step=epoch)
    for name, value in gpu_memory_snapshot().items():
        mlflow.log_metric(name, value, step=epoch)


def _save_and_maybe_upload(
    cfg: Config,
    accelerator: Accelerator,
    checkpoint_root: Path,
    meta: CheckpointMeta,
) -> Path:
    """Save the periodic checkpoint, prune to top-K, and (if configured)
    mirror the latest one to the spot-survival URI. Returns the local path."""
    ckpt_path = checkpoint_root / f"epoch_{meta.epoch:04d}"
    save_checkpoint(accelerator, ckpt_path, meta)
    prune_top_k(checkpoint_root, cfg.trainer.keep_top_k_checkpoints)
    if cfg.trainer.checkpoint_uri is not None:
        # Best-effort: upload failures are logged but don't kill the run.
        try:
            upload_checkpoint(ckpt_path, cfg.trainer.checkpoint_uri)
        except Exception:
            logger.exception("Periodic checkpoint upload failed.")
    return ckpt_path


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
            epoch,
            TRACKED_METRIC_NAME,
            epochs_since_improvement,
        )
        return ("stopped_at_epoch", float(epoch))

    elapsed = time.monotonic() - train_start
    if cfg.trainer.max_wall_seconds is not None and elapsed >= cfg.trainer.max_wall_seconds:
        logger.info(
            "Wall-clock budget exhausted at epoch %d (%.1fs >= %ds). Exiting cleanly.",
            epoch,
            elapsed,
            cfg.trainer.max_wall_seconds,
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
