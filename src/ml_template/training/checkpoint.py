"""Checkpoint save/load on top of Accelerate.

`Accelerator.save_state()` / `load_state()` already handle model + optimizer
+ scheduler + RNG state (torch / cuda / numpy / python). This module wraps
those calls with three things they don't provide:

1. **Atomic writes**: write to a staging directory next to the target,
   then rename. POSIX rename is atomic on the same filesystem so a
   crash mid-save can't leave a half-written checkpoint.
2. **Metadata sidecar**: a small JSON with epoch / step / val metric so
   we can pick the best checkpoint without loading state.
3. **Top-K retention**: keep only the K best by val metric, delete the
   rest. Saves disk and (in the SIGTERM-upload commit) bandwidth.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from accelerate import Accelerator

logger = logging.getLogger(__name__)

META_FILENAME = "checkpoint_meta.json"


@dataclass
class CheckpointMeta:
    """Sidecar metadata persisted alongside Accelerator state."""

    epoch: int
    step: int
    metric_name: str
    metric_value: float
    # "min" → smaller is better (loss); "max" → larger is better (accuracy).
    metric_mode: str

    def is_better_than(self, other: CheckpointMeta) -> bool:
        if self.metric_mode == "min":
            return self.metric_value < other.metric_value
        return self.metric_value > other.metric_value


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    meta: CheckpointMeta,
) -> Path:
    """Save accelerator state + metadata atomically. Returns the final path."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    staging = output_dir.with_name(f".{output_dir.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        accelerator.save_state(str(staging))
        (staging / META_FILENAME).write_text(json.dumps(asdict(meta), indent=2))
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    logger.info(
        "Saved checkpoint: %s (epoch=%d step=%d %s=%.4f)",
        output_dir,
        meta.epoch,
        meta.step,
        meta.metric_name,
        meta.metric_value,
    )
    return output_dir


def load_checkpoint(accelerator: Accelerator, input_dir: Path) -> CheckpointMeta:
    """Restore Accelerator state from `input_dir` and return its metadata."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {input_dir}")
    meta_path = input_dir / META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No {META_FILENAME} in {input_dir} — refusing to resume from an "
            f"unidentified checkpoint."
        )

    accelerator.load_state(str(input_dir))
    meta_dict = json.loads(meta_path.read_text())
    meta = CheckpointMeta(**meta_dict)
    logger.info(
        "Resumed from %s (epoch=%d step=%d %s=%.4f)",
        input_dir,
        meta.epoch,
        meta.step,
        meta.metric_name,
        meta.metric_value,
    )
    return meta


def prune_top_k(checkpoint_root: Path, keep_k: int) -> list[Path]:
    """Keep the K best checkpoints under `checkpoint_root` by val metric.

    Returns the list of directories that were deleted. Treats any
    subdirectory containing `checkpoint_meta.json` as a checkpoint.
    """
    checkpoint_root = Path(checkpoint_root)
    if keep_k <= 0:
        raise ValueError(f"keep_k must be >= 1, got {keep_k}")
    if not checkpoint_root.is_dir():
        return []

    candidates: list[tuple[CheckpointMeta, Path]] = []
    for sub in checkpoint_root.iterdir():
        meta_path = sub / META_FILENAME
        if not (sub.is_dir() and meta_path.is_file()):
            continue
        candidates.append((CheckpointMeta(**json.loads(meta_path.read_text())), sub))

    if len(candidates) <= keep_k:
        return []

    # Sort best-first: for "min" mode ascending, for "max" mode descending.
    # All checkpoints in a single training run share the same metric_mode.
    mode = candidates[0][0].metric_mode
    reverse = mode == "max"
    candidates.sort(key=lambda c: c[0].metric_value, reverse=reverse)

    deleted: list[Path] = []
    for _, path in candidates[keep_k:]:
        shutil.rmtree(path)
        deleted.append(path)
        logger.info("Pruned checkpoint: %s", path)
    return deleted
