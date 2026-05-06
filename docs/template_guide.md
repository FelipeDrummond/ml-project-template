# Template guide

This document explains what this repo *is* as a template — the architecture,
the configuration system, and every feature it ships with. Read it once
when instantiating the template for a new project.

The most important section for cost-sensitive personal projects is
[GPU efficiency features](#gpu-efficiency-features) — every knob is listed
with **when it's wrong for your project** and **what to change first**.
If you're starting a new project, jump to
[Adapting to a new project](#adapting-to-a-new-project) for a per-project-type
checklist.

---

## Philosophy

- **Strict-but-pragmatic** lint, types, env management. Friction with safety
  rails is cheaper than friction with mystery bugs.
- **Cost-conscious** for cloud GPU runs paid out-of-pocket. Every feature
  that touches the training loop has a "what's it doing to my bill?" answer.
- **Agent-first** layout. The Makefile is the public API; CLAUDE.md
  documents where things live; everything important is testable.
- **Minimum surface area**. We deliberately don't ship: DDP/FSDP, ZeRO, custom
  CUDA kernels, ONNX/TensorRT, Optuna sweepers, OOM auto-retry, dataset cache
  layers. They're all good in their place but wrong-shaped for a personal-project
  template — add per-project as needed.

## Architecture

```
src/ml_template/
  config_schemas.py     Hydra structured-config dataclasses (typed; YAML validated against them)
  cli/train.py          Hydra entrypoint: composes config, calls training.train()
  cli/eval.py           Eval entrypoint stub (per-project)
  data/dataset.py       Synthetic placeholder Dataset + build_dataloaders()
  models/mlp.py         Tiny MLP placeholder
  training/loop.py      The main training loop (Accelerate-driven)
  training/checkpoint.py Atomic save / load / top-K retention
  training/scheduler.py LR schedules (linear-warmup-cosine)
  training/spot.py      SIGTERM handler + fsspec checkpoint upload
  training/reproducibility.py Git/env capture + dirty-tree gate
  utils/device.py       Device resolution helper (auto / cuda / mps / cpu)
  utils/seed.py         Reproducibility seeding
  utils/perf.py         TF32 / fused AdamW / NaN guard / GPU memory snapshot
  utils/logging.py      Stdlib logging setup
configs/                Hydra config tree
  config.yaml           Top-level composition
  data/, model/, trainer/  Per-group YAMLs
  experiment/example.yaml   `# @package _global_` overlays
scripts/
  overfit_one_batch.py  Standalone smoke test (no train loop dependency)
  check_device.py       Print torch backends on this host
  check_docs_methods_paths.py  Pre-commit hook: validates docs/methods.md paths
tests/                  Unit + integration tests; mirrors src/
docs/                   Research + project documentation
```

## Configuration system (Hydra)

Configs are typed dataclasses in `src/ml_template/config_schemas.py`,
registered with Hydra's `ConfigStore`. The YAML files in `configs/` extend
those dataclasses. Three layers of validation:

1. **Compose-time** — typos in any YAML or CLI override fail with
   `ConfigCompositionException`.
2. **Load-time** — `OmegaConf.to_object(cfg)` returns a real `Config`
   instance; missing/mistyped fields fail here.
3. **Static parity** — `tests/test_config_schema.py` walks every YAML in
   `configs/` on every CI run and verifies all keys correspond to a
   dataclass field. Catches typos in unused experiment files that would
   otherwise rot undetected.

Override anything from the CLI (no quoting needed for simple values):

```bash
make train OVERRIDES="trainer=cloud trainer.lr=3e-4 model.hidden_dim=256 seed=7"
make train OVERRIDES="experiment=example"            # named composition
make train OVERRIDES="trainer.scheduler=null"        # null literal
make train OVERRIDES="run.allow_dirty=true"          # bypass the dirty-tree gate
```

**Adding a new group** (e.g. a `dataset_v2` config group):

1. Add a `@dataclass class DatasetV2Config` in `config_schemas.py`.
2. Add it as a typed field on `Config` (`dataset_v2: DatasetV2Config = field(default_factory=DatasetV2Config)`).
3. In `register_configs()`, add `cs.store(group="dataset_v2", name="base_dataset_v2", node=DatasetV2Config)`.
4. Add `configs/dataset_v2/<variant>.yaml` files.
5. Update `NESTED_SCHEMAS` in `tests/test_config_schema.py` so static parity covers the new group.

**Gotcha**: Hydra's `+key=val` (with the `+` prefix) explicitly *adds* a new
field and bypasses struct mode. Never use `+` to "fix" a typo — that hides
the typo instead of catching it.

---

## GPU efficiency features

Every knob below has the same shape:
- **What** it does
- **How to configure** (config keys / CLI override)
- **When it's wrong for your project**
- **First thing to change** (if it's wrong, change this *before* training)

### Tier-1 perf knobs (always-on, free)

These are silent no-ops where they don't apply. You almost never need to
disable them.

| Knob | Default | Disabling? |
|---|---|---|
| TF32 | always on (hardcoded in `enable_tf32`) | Silent no-op on non-Ampere. Edit the function if you really need it off. |
| Fused AdamW | auto-on when CUDA | Auto-skipped on MPS/CPU. Leave on. |
| NaN/Inf loss guard | always | ~3 lines, catches divergence in seconds. Leave on. |
| `set_to_none=True` for `zero_grad` | always | Standard. Leave on. |

### Mixed precision (`trainer.precision`)

- **What**: Accelerate's autocast. `"no"` | `"fp16"` | `"bf16"` | `"fp8"`.
- **Configured**: `trainer.precision: bf16` in `configs/trainer/cloud.yaml`,
  `"no"` in `configs/trainer/local.yaml`.
- **When wrong**:
  - **V100, T4, 2080Ti, or any pre-Ampere CUDA GPU** → bf16 is unsupported.
    Switch to `"fp16"`. Note that fp16 needs loss scaling (Accelerate handles
    this internally when you pass `precision="fp16"`).
  - **MPS local** → MPS doesn't autocast cleanly. The training loop
    silently downgrades to `"no"` with a warning logged. No action needed.
  - **Numerically sensitive ops** (softmax over a very large vocab,
    log/exp of small numbers, custom layer norms) → may need fp32
    regions. Currently not exposed — wrap the sensitive op in
    `with torch.autocast(device_type="cuda", enabled=False):` per-project.
- **First thing to change**: if renting V100 / T4 / 2080Ti, set
  `trainer.precision=fp16` in `configs/trainer/cloud.yaml`.

### DataLoader perf

- **What**: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`.
- **Configured**: `configs/data/default.yaml`. `num_workers > 0` is gated on
  `torch.cuda.is_available()` at use site (macOS spawn-based workers usually
  hurt local throughput, so locally we run with `workers=0` regardless).
- **When wrong**:
  - **Tiny / synthetic data fully in memory** → worker overhead dominates;
    `num_workers=0` is faster.
  - **Dataset that does CUDA preprocessing** → workers can't share a CUDA
    context cleanly. Use `num_workers=0` and prep on the main process, or
    pre-compute and cache.
  - **Sim2real / RL with on-policy rollouts** → the simulator is the
    bottleneck and lives on the main process. `num_workers=0`.
- **First thing to change**: `data.num_workers=0` if your data is fast
  (synthetic / in-memory) or does CUDA prep.

### `torch.compile`

- **What**: Whole-graph compile via TorchDynamo. CUDA-only.
- **Configured**: `trainer.compile_mode: null | "default" | "reduce-overhead" | "max-autotune"`.
- **When wrong**:
  - **Dynamic shapes** (variable-length seqs / padding) → graph breaks
    happen often, may slow you down or crash. Keep `null`.
  - **Custom CUDA ops or non-tensor control flow in `forward`** → graph
    breaks. Keep `null`.
  - **NaN / divergence debugging** → compile obscures the offending op.
    Disable while debugging, re-enable after.
  - **Small models (<10M params)** → compile overhead may not pay back.
- **First thing to change**: leave `null` until you've trained one
  un-compiled run end-to-end. Then flip to `"default"`.

### LR scheduler (`trainer.scheduler`)

- **What**: Linear-warmup-then-cosine. `null` = constant LR.
- **Configured**: `linear_warmup_cosine` in `configs/trainer/cloud.yaml`,
  `null` in `configs/trainer/local.yaml`.
- **When wrong**:
  - **RL (PPO, SAC, anything online-policy)** → the convention is constant
    LR. Cosine decay can collapse the policy. **Set `scheduler: null`.**
  - **Fine-tuning a pretrained model** → you usually want a much smaller
    peak LR and a different curve (e.g. linear-only). The current schedule
    works but isn't optimal.
  - **Total optim-step count is data-driven and unknown ahead of time**
    (e.g. `early_stop_patience` cuts you off mid-cosine) → benign; the
    schedule simply doesn't reach the floor.
- **First thing to change**: `trainer.scheduler=null` if doing RL.

### Gradient accumulation (`trainer.grad_accum_steps`)

- **What**: Accumulate gradients across N micro-batches, step the
  optimizer once. Effective batch = `data.batch_size × grad_accum_steps`.
- **Configured**: `trainer.grad_accum_steps: 1` (no accumulation) by default.
- **When wrong**:
  - **Your model uses BatchNorm** → BN running stats are computed
    per-micro-batch, which is wrong under accumulation. **Switch BN to
    GroupNorm / LayerNorm / RMSNorm, or use SyncBatchNorm**, *before*
    bumping `grad_accum_steps`.
  - **Already at the largest batch size your GPU supports** → just costs
    more wall-clock with no benefit.
- **First thing to change**: if your model has BatchNorm, swap norm layers
  before increasing `grad_accum_steps`.

### Cost controls

- **What**: `early_stop_patience` (stop when val/loss plateaus N epochs)
  and `max_wall_seconds` (hard budget). Both `null` disables.
- **When wrong**:
  - **RL / noisy reward signals** → patience semantics break (val/loss
    isn't monotonic in expectation). **Set `early_stop_patience: null`.**
  - **No budget constraint** → leave `max_wall_seconds: null`.
- **First thing to change**: `trainer.early_stop_patience=null` if doing RL.

### Tracked metric (currently `val/loss`, min mode)

- **What**: The metric that decides "best checkpoint" for top-K retention,
  early stopping, and the SIGTERM-saved checkpoint.
- **Configured**: hard-coded constants `TRACKED_METRIC_NAME` and
  `TRACKED_METRIC_MODE` at the top of `src/ml_template/training/loop.py`.
  Not in the config schema (yet).
- **When wrong**:
  - **You care about val/accuracy or val/iou or val/return** → edit those
    two constants.
  - **You don't have a clean val set** (RL again) → consider tracking
    `train/loss` instead, or disable early-stop and pick top-K by epoch.
- **First thing to change**: if your "best" isn't val/loss-min, edit
  `loop.py:TRACKED_METRIC_NAME` and `TRACKED_METRIC_MODE` (`"min"` or `"max"`).

### Checkpoints (atomic save, top-K retention, resume)

- **What**: Per-epoch checkpoint via `accelerator.save_state()` (model +
  optim + scheduler + RNG + dataloader sampler state). Atomic via staging
  dir + rename. Top-K best by tracked metric.
- **Configured**: `trainer.checkpoint_every_n_epochs`,
  `trainer.keep_top_k_checkpoints`, `trainer.resume_from`.
- **When wrong**:
  - **Want every checkpoint for retroactive analysis** → set
    `keep_top_k_checkpoints` to a large number (or `total_epochs`).
  - **Disk pressure on the cloud box** → bump `checkpoint_every_n_epochs`
    so you write less often, or reduce `keep_top_k_checkpoints`.
- **First thing to change**: usually nothing.

### Spot-instance survival (`trainer.checkpoint_uri`)

- **What**: SIGTERM handler catches preempt, saves a last-gasp local
  checkpoint, then uploads to the configured fsspec URI (`s3://`,
  `gs://`, `file://`). Periodic checkpoints also mirror to the URI.
- **Configured**: `trainer.checkpoint_uri` (commonly via env:
  `CHECKPOINT_URI=...`). Install the matching backend extra:
  `uv sync --extra dev --extra cuda --extra s3` (or `--extra gcs`).
- **When wrong**:
  - **Not on a spot instance** → leave `null`. Local checkpoints survive
    on the persistent disk anyway.
  - **Slow upload bandwidth** → adds latency per epoch. Bump
    `checkpoint_every_n_epochs` so you upload less often.
- **First thing to change**: leave `null` unless you're explicitly on
  spot/preemptible.

### `fast_dev_run` + CI gate

- **What**: 1 train batch + 1 val batch + 1 forced checkpoint
  save→load round-trip, end-to-end. Runs as a CI step on every PR.
- **Configured**: `trainer.fast_dev_run: true` from CLI, or
  `make dev-run`.
- **What it doesn't catch** (yet):
  - bf16-specific or `torch.compile`-specific bugs (CI runs CPU only)
  - DDP / FSDP / multi-GPU specifics
  - Long-tail numerical issues that only show up after many epochs
- **First thing to change**: nothing — runs automatically. Re-run locally
  with `make dev-run` after any model/data/scheduler change.

### Observability (MLflow)

- **What's logged**:
  - **Per step**: `train/loss`, `train/lr`
  - **Per epoch**: `val/loss`, `val/accuracy`, `perf/samples_per_sec`,
    `perf/dataloader_wait_pct`, `perf/epoch_seconds`,
    `gpu/mem_alloc_mib`, `gpu/mem_reserved_mib`, `gpu/mem_peak_mib`,
    `gpu/util_pct` (when pynvml available)
  - **Per run**: full config, git SHA/branch/dirty, env summary, argv
  - **Artifacts**: `git_diff.patch` (when dirty), `packages.txt`, `resolved_config.txt`
- **How to read**:
  - `perf/dataloader_wait_pct > 5%` → IO bound. Bump `num_workers`,
    `prefetch_factor`, or shrink per-sample preprocessing.
  - `gpu/util_pct = 100%` does **NOT** mean the GPU is saturated. It means
    "≥1 kernel was running". Tiny back-to-back kernels show 100% util while
    doing little useful work. The proper saturation metric is MFU; we
    chose util% for simplicity.
- **First thing to change**: nothing — read it.

### Reproducibility envelope

- **What**: Strict-by-default refuse-dirty gate. Every run logs the full
  reproducibility envelope to MLflow (see [observability](#observability-mlflow)
  above).
- **Configured**: `run.allow_dirty: false` (default) blocks training when
  the working tree has uncommitted changes. `run.allow_dirty=true` is the
  per-run escape hatch.
- **When wrong**:
  - **Fast iteration with always-dirty tree** → either commit-as-you-go
    (preferred), `git stash` your changes, or pass
    `run.allow_dirty=true` per CLI invocation.
  - **No git repo** → silently no-op.
- **First thing to change**: nothing. If the strict default is too
  intrusive day-to-day, run `make dev-run` instead of `make train` while
  iterating (dev-run also goes through the gate but you typically commit
  before a real training run anyway).

---

## Adapting to a new project

After `gh repo create <your-project> --template FelipeDrummond/ml-project-template`:

### Always (first 30 minutes)

1. Rename the package directory: `src/ml_template/` → `src/<your_project>/`.
2. Update imports across `src/` and `tests/`. Find/replace `ml_template` → `<your_project>`.
3. Update `pyproject.toml`: `[project].name`, `[project.scripts]`,
   `[tool.hatch.build.targets.wheel].packages`, `[tool.coverage.run].source`.
4. Update `Makefile`: `python -m ml_template.cli.train` →
   `python -m <your_project>.cli.train`.
5. Replace `docs/research_statement.md`, `docs/data_card.md`,
   `docs/model_card.md` stubs with your project's content.
6. Replace `src/<your_project>/data/dataset.py` and
   `src/<your_project>/models/mlp.py` with the real things.
7. Run `make dev-run` to verify the pipeline works end-to-end.

### By project type

#### Behavior cloning / supervised / imitation learning (template's sweet spot)

Defaults are mostly right. Tune as you go:
- Adjust `trainer.warmup_steps` to dataset size — small datasets benefit
  from short warmup (50-200 steps); huge datasets cap out at 1000 by default.
- Set `trainer.compile_mode="default"` once you've trained one un-compiled run.

#### Reinforcement learning (PPO, SAC, online RL)

Required deviations:
1. **`trainer.scheduler: null`** (constant LR is the convention).
2. **`trainer.early_stop_patience: null`** (reward noise breaks patience).
3. **Tune `trainer.grad_clip_max_norm`** per algorithm. PPO commonly uses
   `0.5`; SAC sometimes `1.0` or higher; tune for your env.
4. Reconsider `trainer.precision`. RL is often numerically sensitive —
   start at `"no"`, switch to `"bf16"` only after validating.
5. Replace `data/dataset.py` with your env rollout buffer (or its equivalent).
6. Probably edit `loop.py:TRACKED_METRIC_NAME` to `"val/return"` (or whatever
   your evaluation rollout reports) with `TRACKED_METRIC_MODE = "max"`.

#### Vision / CNNs

If you'll use `grad_accum_steps > 1` and your model has BatchNorm:
1. **Switch BN → GroupNorm / LayerNorm / RMSNorm**, OR use
   `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`, BEFORE bumping
   `grad_accum_steps`. Otherwise BN running stats will be silently wrong.

Otherwise defaults work. Watch for `data.pin_memory` benefits — they're
real for vision but only when you're CUDA-bound.

#### Transformers / language models

Defaults are good. After validation:
1. Set `trainer.compile_mode="default"` for ~30-100% speedup.
2. If your batches are padded variable-length, `compile` may break with
   graph breaks — keep `null` and benchmark.
3. Consider bumping `trainer.grad_accum_steps` to reach effective batch
   sizes that don't fit in one GPU (works without code changes since we
   use `accelerator.accumulate()`).

#### Robotics (sim2real, online learning)

Same as RL above, plus:
1. **`data.num_workers: 0`** since the simulator is the bottleneck and
   lives on the main process.
2. **`trainer.max_wall_seconds`** is often useful for hard budget caps on
   long online runs.

---

## Operations cheat sheet

```bash
# Install
make install              # Mac MPS / CPU
make install-cuda         # Linux CUDA 12.1

# Develop
make smoke                # Standalone overfit-1-batch (broken model? this catches it)
make dev-run              # fast_dev_run end-to-end on CPU
make check                # ruff + pyright + pytest
make format               # ruff format + autofix

# Train
make train                                              # local profile
make train OVERRIDES="trainer=cloud"                    # cloud profile
make train OVERRIDES="trainer.compile_mode=default"     # one-off override
make train OVERRIDES="run.allow_dirty=true"             # bypass dirty-tree gate

# Cloud sync (set REMOTE and REMOTE_DIR first)
make cloud-setup          # ssh remote, git pull, install-cuda
make cloud-train          # ssh remote, run training
make pull-results         # rsync mlruns/ from remote
```

## What to read next

- **`CLAUDE.md`** — agent-facing index of the repo (where things live, the verification protocol, house rules).
- **`docs/research_statement.md`** — replace with your project's goal.
- **`docs/methods.md`** — the architecture / training-procedure doc, kept in sync with code by a pre-commit hook.
- **`tests/test_config_schema.py`** — pins the config validation contract; if you add a new config group, update `NESTED_SCHEMAS` here.
