"""Tests for the reproducibility envelope: git introspection, env capture,
dirty-tree gate."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mlflow
import pytest

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    RunConfig,
    TrainerConfig,
)
from ml_template.training import train
from ml_template.training.reproducibility import (
    DirtyTreeError,
    assert_clean_or_allowed,
    env_summary,
    freeze_packages,
    git_diff,
    git_state,
)

# ---------------------------------------------------------------------------
# Pure-function tests against a sandboxed git repo.
# ---------------------------------------------------------------------------


def _make_repo(root: Path, dirty: bool = False) -> Path:
    """Build a single-commit git repo at `root`. Optionally leave it dirty."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.email", "t@t"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "t"], cwd=root)
    subprocess.check_call(["git", "config", "commit.gpgsign", "false"], cwd=root)
    (root / "a.txt").write_text("hello")
    subprocess.check_call(["git", "add", "."], cwd=root)
    subprocess.check_call(["git", "commit", "-q", "-m", "init"], cwd=root)
    if dirty:
        (root / "a.txt").write_text("modified!")
    return root


def test_git_state_in_clean_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path / "repo", dirty=False)
    monkeypatch.chdir(repo)
    state = git_state()
    assert state["git/sha"]  # 40-char SHA
    assert len(state["git/sha"]) == 40
    assert state["git/dirty"] == "false"


def test_git_state_in_dirty_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path / "repo", dirty=True)
    monkeypatch.chdir(repo)
    state = git_state()
    assert state["git/dirty"] == "true"
    assert git_diff() != ""


def test_git_state_outside_a_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = git_state()
    assert state["git/sha"] == ""
    assert state["git/dirty"] == "false"


def test_assert_clean_or_allowed_passes_when_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo(tmp_path / "repo", dirty=False)
    monkeypatch.chdir(repo)
    assert_clean_or_allowed(allow_dirty=False)  # no exception


def test_assert_clean_or_allowed_blocks_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo(tmp_path / "repo", dirty=True)
    monkeypatch.chdir(repo)
    with pytest.raises(DirtyTreeError, match="uncommitted changes"):
        assert_clean_or_allowed(allow_dirty=False)


def test_assert_clean_or_allowed_passes_when_dirty_with_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo(tmp_path / "repo", dirty=True)
    monkeypatch.chdir(repo)
    assert_clean_or_allowed(allow_dirty=True)  # no exception


def test_assert_clean_or_allowed_passes_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No git repo → nothing to enforce, must not raise."""
    monkeypatch.chdir(tmp_path)
    assert_clean_or_allowed(allow_dirty=False)


def test_env_summary_has_required_keys() -> None:
    summary = env_summary()
    assert summary["env/torch"]
    assert summary["env/python"]
    assert summary["env/hostname"]
    assert summary["env/cuda_available"] in {"True", "False"}


def test_freeze_packages_returns_something() -> None:
    """uv/pip is available in our dev env; the freeze is non-empty."""
    if shutil.which("uv") is None and shutil.which("pip") is None:
        pytest.skip("Neither uv nor pip available.")
    out = freeze_packages()
    # Should at least mention torch (or accelerate).
    assert "torch" in out or "accelerate" in out


# ---------------------------------------------------------------------------
# End-to-end: training logs the envelope to MLflow.
# ---------------------------------------------------------------------------


def test_training_logs_reproducibility_envelope(tmp_path: Path) -> None:
    cfg = Config(
        seed=0,
        output_dir=str(tmp_path / "run"),
        run=RunConfig(allow_dirty=True),  # the autouse fixture stubs the gate
        data=DataConfig(n_samples=32, n_features=4, n_classes=2, batch_size=4),
        model=ModelConfig(hidden_dim=8, n_layers=1),
        trainer=TrainerConfig(device="cpu", epochs=1, lr=1e-3, grad_clip_max_norm=None),
        mlflow=MLflowConfig(
            tracking_uri=f"file:{tmp_path / 'mlruns'}",
            experiment_name="repro_envelope",
        ),
    )
    train(cfg)

    client = mlflow.MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name("repro_envelope")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    params = runs[0].data.params
    # Git info present (may be empty strings if not run in a repo, but keys exist).
    assert "git/sha" in params
    assert "git/dirty" in params
    # Env keys.
    assert "env/torch" in params
    assert "env/python" in params
    assert "env/hostname" in params
    # CLI argv.
    assert "run/argv" in params

    # Artifacts: packages.txt and resolved_config.txt always; git_diff.patch
    # only when dirty (we don't assert on that here).
    artifact_uri = runs[0].info.artifact_uri
    assert artifact_uri is not None
    artifact_dir = Path(artifact_uri.removeprefix("file://"))
    assert (artifact_dir / "packages.txt").is_file()
    assert (artifact_dir / "resolved_config.txt").is_file()
