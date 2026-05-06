"""Reproducibility envelope: git state, env capture, dirty-tree gate.

The "what was actually in this run?" question is the single most common
six-months-later question. Capture enough at run start that you can
answer it without going hunting:

- Git SHA, branch, dirty flag, **full diff as MLflow artifact** when dirty
- `sys.argv`, full resolved Hydra config (post-overrides)
- `uv pip freeze` (or `pip freeze` fallback) as artifact
- torch/CUDA/device/python/hostname

Strict-by-default: a dirty working tree refuses to run. Override per-run
with `run.allow_dirty=true`. Lab-internal templates (Meta, DeepMind)
typically enforce this — the cost of one mystery run dwarfs the
annoyance of either committing or passing the flag.
"""

from __future__ import annotations

import logging
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class DirtyTreeError(RuntimeError):
    """Raised when training would start on uncommitted code without override."""


def _git(args: list[str], cwd: Path | None = None) -> str:
    """Run a git command. Returns stdout stripped, or empty string on failure."""
    if shutil.which("git") is None:
        return ""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def git_state() -> dict[str, str]:
    """SHA / branch / dirty flag. Empty values when not in a git repo."""
    sha = _git(["rev-parse", "HEAD"])
    if not sha:
        return {"git/sha": "", "git/branch": "", "git/dirty": "false"}
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _git(["status", "--porcelain"])
    return {
        "git/sha": sha,
        "git/branch": branch,
        "git/dirty": "true" if porcelain else "false",
    }


def git_diff() -> str:
    """Full working-tree diff (staged + unstaged), suitable as an artifact."""
    return _git(["diff", "HEAD"])


def env_summary() -> dict[str, str]:
    """Environment fingerprint we care about for ML reproducibility."""
    summary = {
        "env/python": sys.version.split()[0],
        "env/platform": platform.platform(),
        "env/hostname": socket.gethostname(),
        "env/torch": torch.__version__,
        "env/cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        # `torch.version.cuda` exists at runtime but isn't in the public stub.
        cuda_ver = getattr(torch.version, "cuda", "unknown")  # pyright: ignore[reportAttributeAccessIssue]
        summary["env/cuda_version"] = str(cuda_ver)
        summary["env/gpu"] = torch.cuda.get_device_name(0)
    return summary


def freeze_packages() -> str:
    """Best-effort `uv pip freeze`, falling back to `pip freeze`."""
    for cmd in (["uv", "pip", "freeze"], [sys.executable, "-m", "pip", "freeze"]):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except subprocess.CalledProcessError:
            continue
    return ""


def assert_clean_or_allowed(allow_dirty: bool) -> None:
    """Refuse to start training when the working tree is dirty.

    Override with `run.allow_dirty=true` (intentional escape hatch) or
    by committing/stashing your changes (the right answer most times).
    """
    state = git_state()
    if state.get("git/sha", "") == "":
        # Not in a git repo — nothing to enforce.
        return
    if state["git/dirty"] == "true" and not allow_dirty:
        raise DirtyTreeError(
            "Working tree has uncommitted changes — refusing to start training. "
            "Either commit/stash your changes (preferred) or pass "
            "`run.allow_dirty=true` (escape hatch). The diff would otherwise "
            "be lost from the run record. Run `git status` for details."
        )
