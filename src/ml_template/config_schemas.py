"""Hydra structured configs.

Registering dataclasses with Hydra's `ConfigStore` lets us type-check the
loaded config: typos and bad types fail at load, not three minutes into a
training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    n_samples: int = 256
    n_features: int = 16
    n_classes: int = 4
    batch_size: int = 32
    val_fraction: float = 0.2
    # DataLoader workers + perf flags. `num_workers > 0` is gated on CUDA
    # availability at use site: spawn-based workers on macOS often hurt
    # throughput, so locally we run with workers=0 regardless.
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.0


@dataclass
class TrainerConfig:
    # device: one of "auto" | "cuda" | "mps" | "cpu". `cpu` is a hard
    # override (passed as `cpu=True` to Accelerator); the others let
    # Accelerate auto-detect (CUDA > MPS > CPU). Validated at use site.
    device: str = "auto"
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    log_every_n_steps: int = 10
    # Perf knobs (Tier 1).
    tf32: bool = True
    # Gradient clipping is regularization / training-stability, not a NaN
    # guard (those are separate). `null` disables clipping.
    grad_clip_max_norm: float | None = 1.0
    # Dev-only: torch.autograd.set_detect_anomaly is very slow; enable to
    # locate the op producing NaN/Inf, then turn off.
    detect_anomaly: bool = False
    # Mixed precision via Accelerate: "no" | "fp16" | "bf16" | "fp8".
    # Default is "no" so local dev (Mac MPS / CPU) doesn't autocast.
    # Use "bf16" on Ampere+ cloud GPUs.
    precision: str = "no"
    # torch.compile flag. CUDA-only (silently disabled elsewhere). Mode:
    # "default" | "reduce-overhead" | "max-autotune". `null` disables.
    # First call is slow (compilation); subsequent steps are 30-100%
    # faster on most workloads. Worth a flag, opt-in by default since
    # compile can shadow real bugs (e.g. graph breaks at NaN).
    compile_mode: str | None = None
    # Checkpointing.
    checkpoint_every_n_epochs: int = 1
    keep_top_k_checkpoints: int = 3
    # Path to a checkpoint directory to resume from. When set, training
    # restores model, optimizer, and RNG state, then continues from the
    # next epoch. `null` starts from scratch.
    resume_from: str | None = None
    # Cost controls — both null disables.
    # Stop training when val/loss hasn't improved for N consecutive epochs.
    early_stop_patience: int | None = None
    # Hard wall-clock budget. Run exits cleanly (after saving the current
    # checkpoint) once exceeded. Useful for capping spot-instance spend.
    max_wall_seconds: int | None = None


@dataclass
class MLflowConfig:
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "default"
    run_name: str | None = None


@dataclass
class Config:
    seed: int = 42
    # Hydra populates this from `HydraConfig.get().runtime.output_dir` in
    # cli/train.py — used as the root for checkpoints. Tests pass an
    # explicit path.
    output_dir: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


def register_configs() -> None:
    """Register the config schemas with Hydra. Call at CLI entry."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="trainer", name="base_trainer", node=TrainerConfig)
