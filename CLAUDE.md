# CLAUDE.md

Agent-facing map of this repo. Read this first.

## Overview

PyTorch deep-learning research project. Local dev on MacBook M-series (MPS or
CPU); real training on cloud Linux + CUDA 12.1. See `README.md` §"Hardware
split" for the contract — the two environments are not interchangeable.

## Hardware contract

| | Local | Cloud |
|---|---|---|
| Backend | MPS / CPU | CUDA |
| Used for | dev, smoke, single-batch overfit | full training, sweeps |

**Hard rule:** never claim a CUDA training run was executed unless it actually
ran. If you cannot run on CUDA from your environment, say so — do not
fabricate metrics, loss curves, or run ids.

## Where things live

| Concern | Path |
|---|---|
| Training entrypoint | `src/ml_template/cli/train.py` |
| Eval entrypoint | `src/ml_template/cli/eval.py` |
| Training loop | `src/ml_template/training/loop.py` |
| Model definitions | `src/ml_template/models/` |
| Data + dataloaders | `src/ml_template/data/dataset.py` |
| Device resolution | `src/ml_template/utils/device.py` |
| Seeding | `src/ml_template/utils/seed.py` |
| Logging setup | `src/ml_template/utils/logging.py` |
| Typed configs (Hydra) | `src/ml_template/config_schemas.py` |
| YAML configs | `configs/` |
| Local trainer profile | `configs/trainer/local.yaml` |
| Cloud trainer profile | `configs/trainer/cloud.yaml` |
| Smoke test (script) | `scripts/overfit_one_batch.py` |
| Smoke test (pytest) | `tests/test_train_smoke.py` |
| Tests | `tests/` |
| Research docs | `docs/` (start at `docs/README.md`) |

## Commands

The Makefile is the source of truth for "what commands exist." Prefer Make
targets over ad-hoc Python invocations.

```bash
make help          # list targets
make install       # local Mac env (MPS/CPU)
make install-cuda  # cloud CUDA env
make lint
make format
make typecheck
make test
make smoke         # one-batch overfit; must drive loss → ~0
make train         # default Hydra config
make train OVERRIDES="trainer=cloud model.hidden_dim=256"
make check         # lint + typecheck + test (the standard pre-push check)
```

For cloud workflow targets (`cloud-setup`, `cloud-train`, `pull-results`),
`REMOTE` and `REMOTE_DIR` must be exported.

## Verification protocol

When you change anything in the training path (data, model, loop, configs),
run **in this order** and report exact output:

1. `make lint`
2. `make typecheck`
3. `make test`
4. `make smoke`

If any step fails, stop and surface the failure — do not paper over it. If
the change is to the CUDA-only path and you cannot run it locally, say so
explicitly; the user runs cloud verification.

## House rules

These mirror the user's global guidelines (`~/.claude/CLAUDE.md`). Quick
checklist before submitting a change:

- **Shape comments at boundaries.** When tensors flow between modules, add a
  `# shape: (B, ...)` comment if not obvious.
- **Eval-mode discipline.** `model.eval()` + `torch.inference_mode()` (or
  `torch.no_grad()`) for any non-training pass.
- **Device explicit.** Resolve via `resolve_device(...)`. Never assume
  tensors are on the same device — `.to(device)` at the boundary.
- **Seed early.** Call `set_seed(...)` before any RNG-using code.
- **No `print()`.** Use the `logging` module via `setup_logging()`.
- **No fit-on-full-data.** Scalers, encoders, and statistics are fit on
  **train only**, never on the full dataset before splitting.
- **Explicit dtypes.** Don't let `float64` sneak in at boundaries.
- **Type hints on `src/`.** pyright is strict on `src/`. Annotate function
  signatures.
- **Documents track code.** When you change architecture / loss / training
  procedure in `src/`, update `docs/methods.md` in the same change. The
  pre-commit hook will reject commits whose `docs/methods.md` references
  paths that don't exist.
- **Experiment records.** When a tracked run executes (not a smoke test —
  a real run on real data), create `docs/experiments/EXP-NNNN-<slug>.md`
  and link the MLflow run id. Never fabricate.

## Configs and type checking

The Hydra config tree is type-checked against the dataclasses in
`src/ml_template/config_schemas.py`. Three layers:

1. **Compose-time** — typos in YAML group files and CLI overrides like
   `data.foo=1` (where `foo` doesn't exist on `DataConfig`) fail with a
   `ConfigCompositionException`.
2. **Load-time** — `OmegaConf.to_object(cfg)` in `cli/train.py` returns a real
   `Config` instance; missing or mistyped fields fail here.
3. **Runtime** — `resolve_device(prefer)` validates against `VALID_PREFS`
   (OmegaConf doesn't fully support `Literal`).

Tests in `tests/test_config_schema.py` pin this contract — do not weaken
them without surfacing the change.

**Gotcha**: `+key=val` (with the `+` prefix) explicitly *adds* a new field
and bypasses struct mode. Use it consciously when adding a field; never use
`+` to "fix" a typo that the schema rejects — that hides the typo instead.

**Adding a new config group** (e.g. `scheduler`):
1. Add a `@dataclass class SchedulerConfig` in `config_schemas.py`.
2. Add it as a typed field on `Config`: `scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)`.
3. Register the per-group schema in `register_configs()`:
   `cs.store(group="scheduler", name="base_scheduler", node=SchedulerConfig)`.
4. Create `configs/scheduler/<variant>.yaml` files.
5. Reference the group in `configs/config.yaml` `defaults:` list.

## Reproducibility

Every training run logs to MLflow:
- **Params**: git SHA, branch, dirty flag, full env summary (python, torch,
  CUDA, GPU, hostname), `sys.argv`.
- **Artifacts**: `git_diff.patch` (only when dirty), `packages.txt`
  (`uv pip freeze`), `resolved_config.txt` (the post-Hydra-overrides config
  that actually ran).

**Strict-by-default dirty-tree gate**: `train()` refuses to start when the
working tree has uncommitted changes. Override per-run with
`run.allow_dirty=true` (escape hatch for fast iteration). Implemented in
`src/ml_template/training/reproducibility.py::assert_clean_or_allowed`.
Tests bypass this via an autouse fixture in `tests/conftest.py` — the
gate itself is exercised by `tests/test_reproducibility.py`.

## Cost controls and spot survival

The training loop ships with several knobs aimed at keeping cloud GPU
spend bounded:

- `trainer.precision` — `"bf16"` on Ampere+ for ~2x speedup
- `trainer.tf32` — free ~30% speedup for fp32 ops outside autocast
- `trainer.compile_mode` — opt-in `torch.compile` (CUDA-only)
- `trainer.early_stop_patience` — quit when val/loss plateaus
- `trainer.max_wall_seconds` — hard wall-clock budget
- `trainer.grad_clip_max_norm` — regularization, not a NaN guard (NaN guard is separate)
- `trainer.checkpoint_uri` — fsspec URI (`s3://`, `gs://`, `file://`) where
  every periodic + SIGTERM checkpoint is mirrored. Survives spot reclaim.
  Set via env var: `CHECKPOINT_URI=...` then `trainer.checkpoint_uri=$CHECKPOINT_URI`.
- `trainer.resume_from` accepts both local paths and fsspec URIs.

The SIGTERM handler always runs (regardless of `checkpoint_uri`); it
saves a last-gasp checkpoint locally and uploads to the URI if set.
AWS spot gives 2 min, GCP 30s — uploads are best-effort with timeout
falling through to local.

## What NOT to touch without asking

These have out-of-repo consequences. Stop and confirm with the user before
modifying:

- `pyproject.toml` torch / torchvision versions and the `[tool.uv.sources]` /
  `[tool.uv.index]` blocks — changing them breaks reproducibility on cloud.
- `.github/workflows/ci.yml` — affects what's enforced on PRs.
- `.pre-commit-config.yaml` — affects what's enforced on commit.
- Anything inside `mlruns/` (read-only artifacts of past runs).

## Renaming the package

When this template is instantiated for a new project, rename `ml_template`
to your project's package name. Files to update:

- Directory: `src/ml_template/` → `src/<new_name>/`
- `pyproject.toml`: `[project].name`, `[project.scripts]`, `[tool.hatch.build.targets.wheel].packages`, `[tool.coverage.run].source`
- All `from ml_template.` imports across `src/` and `tests/`
- `Makefile`: the `train` target's `python -m ml_template.cli.train`
- This file (CLAUDE.md): the path table above
