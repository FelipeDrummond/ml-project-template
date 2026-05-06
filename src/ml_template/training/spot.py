"""Spot-interruption survival: SIGTERM handler + fsspec checkpoint upload.

When a spot/preemptible instance is reclaimed, the kernel sends SIGTERM
(AWS gives 2 min, GCP 30s) before SIGKILL. We catch SIGTERM, fast-save
a checkpoint to a remote URI (`CHECKPOINT_URI`), and exit cleanly so
the next run can `trainer.resume_from=<remote_uri>`.

`fsspec` makes the URI scheme generic: `s3://...`, `gs://...`,
`file:///...`, etc. The corresponding backend (`s3fs`, `gcsfs`) ships
as an optional extra; install only the one you need.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import signal
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import fsspec
from accelerate import Accelerator

from ml_template.training.checkpoint import CheckpointMeta, save_checkpoint

logger = logging.getLogger(__name__)


class _CheckpointTaker(Protocol):
    """Callback that produces a CheckpointMeta for the *current* state."""

    def __call__(self) -> CheckpointMeta: ...


def upload_checkpoint(local_dir: Path, remote_uri: str) -> str:
    """Upload `local_dir` to `remote_uri` via fsspec. Returns the remote URI.

    The upload is "atomic-ish": we write to `<remote_uri>.staging.<pid>`
    first, then mv to the final URI. Cloud filesystems vary in their
    rename semantics — S3 in particular is copy+delete, not atomic — but
    the staging path at least prevents partial reads from a parallel job
    seeing a half-uploaded checkpoint.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local checkpoint dir does not exist: {local_dir}")
    fs, root = fsspec.core.url_to_fs(remote_uri)
    staging = f"{root}.staging.{Path(local_dir).name}"
    final = root

    # Wipe any leftover staging from a previous failed upload.
    if fs.exists(staging):
        fs.rm(staging, recursive=True)

    logger.info("Uploading %s -> %s (staging: %s)", local_dir, remote_uri, staging)
    fs.put(str(local_dir) + "/", staging, recursive=True)

    if fs.exists(final):
        fs.rm(final, recursive=True)
    fs.mv(staging, final, recursive=True)
    logger.info("Upload complete: %s", remote_uri)
    return remote_uri


@contextlib.contextmanager
def install_sigterm_handler(
    accelerator: Accelerator,
    checkpoint_uri: str | None,
    take_checkpoint: _CheckpointTaker,
    output_root: Path,
) -> Iterator[None]:
    """Install a SIGTERM handler for the duration of the `with` block.

    On SIGTERM:
    1. Take a fresh checkpoint of the current state via `take_checkpoint()`.
    2. Save it locally under `output_root / "sigterm_ckpt"`.
    3. If `checkpoint_uri` is set, upload via fsspec.
    4. Re-raise SystemExit so the process exits with a non-zero code,
       letting the orchestrator know the run was preempted.

    `take_checkpoint` is a closure the training loop provides — it
    captures the current epoch / step / val metric from the loop's
    locals so we don't have to plumb mutable state into here.
    """
    output_root = Path(output_root)

    def handler(signum: int, _frame: object) -> None:
        logger.warning("Caught signal %d; saving SIGTERM checkpoint.", signum)
        try:
            meta = take_checkpoint()
            local = output_root / "sigterm_ckpt"
            save_checkpoint(accelerator, local, meta)
            if checkpoint_uri is not None:
                try:
                    upload_checkpoint(local, checkpoint_uri)
                except Exception:
                    # Upload failure is logged but not fatal — the local
                    # checkpoint may still be recoverable from the host's
                    # disk if it survives.
                    logger.exception("SIGTERM checkpoint upload failed.")
        finally:
            # SIGTERM: exit 143 (128 + 15) so orchestrators see preemption.
            raise SystemExit(143)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def maybe_resolve_remote_resume(resume_from: str) -> Path:
    """If `resume_from` is a remote fsspec URI, mirror it to a local
    cache and return the local path. Plain filesystem paths pass through.

    Lets `trainer.resume_from=s3://bucket/run-42/sigterm_ckpt` work
    transparently.
    """
    if "://" not in resume_from or resume_from.startswith("file://"):
        return Path(resume_from.removeprefix("file://"))

    fs, remote = fsspec.core.url_to_fs(resume_from)
    cache_root = Path(tempfile.mkdtemp(prefix="ml_template_resume_"))
    local = cache_root / Path(remote).name
    logger.info("Mirroring remote checkpoint %s -> %s", resume_from, local)
    if local.exists():
        shutil.rmtree(local)
    fs.get(remote, str(local), recursive=True)
    return local
