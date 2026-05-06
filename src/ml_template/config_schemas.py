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
    # device: one of "auto" | "cuda" | "mps" | "cpu". Validated at runtime by
    # `resolve_device` (OmegaConf structured configs don't fully support Literal).
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


@dataclass
class MLflowConfig:
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "default"
    run_name: str | None = None


@dataclass
class Config:
    seed: int = 42
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
