# ml-template

PyTorch deep-learning research template for personal projects. Optimized for
**Mac local development → CUDA cloud training** and for development driven by
coding agents (Claude Code).

> Replace this paragraph with a one-line description of your project, then
> fill in `docs/research_statement.md`.

---

## Hardware split — read this first

This template assumes a **two-environment workflow**. They are not
interchangeable; commands differ; lockfile resolution differs.

| | Local (dev) | Cloud (training) |
|---|---|---|
| Hardware | MacBook Pro M-series, Apple Silicon | Linux + NVIDIA GPU |
| Torch backend | MPS (default PyPI wheel covers this) or CPU | CUDA 12.1 |
| Used for | code, debugging, smoke tests, single-batch overfit | full training, sweeps, eval at scale |
| **Not used for** | full training | day-to-day dev |
| Install | `make install` | `make install-cuda` |

Full training runs are **not** done on the Mac. The Mac is for fast iteration
and verification; the cloud GPU is where real training happens.

---

## Setup — local (Mac)

Prereqs: [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv python install 3.11
make install        # installs core deps + dev tools; default torch wheel is MPS-capable
make smoke          # one-batch overfit smoke test — must pass
make check          # lint + types + tests
```

Copy `.env.example` to `.env` and fill in MLflow / remote vars as needed.

## Setup — cloud (CUDA)

On the cloud GPU box (Linux, CUDA 12.1+ available):

```bash
git clone <this-project-url>
cd <project>
make install-cuda   # uv sync --extra dev --extra cuda
make train OVERRIDES="trainer=cloud"
```

`pyproject.toml` routes torch wheels via `[tool.uv.sources]` — the `cuda` extra
selects the `https://download.pytorch.org/whl/cu121` index; the `cpu` extra
selects the CPU index. macOS resolves to the default PyPI wheel regardless
(MPS-capable), so no extra is needed locally.

---

## Daily workflow

Local (laptop):

```bash
# code, then before pushing:
make format         # ruff format + autofix
make check          # ruff check + pyright + pytest (must be green)
make smoke          # if you touched the training path
git push
```

Cloud (run from laptop, controls the remote via SSH):

```bash
export REMOTE=user@cloudbox
export REMOTE_DIR=/workspace/<project>

make cloud-setup    # ssh remote, git pull, install-cuda
make cloud-train    # ssh remote, run training with trainer=cloud
make pull-results   # rsync mlruns/ from remote into local mlruns/
mlflow ui           # browse runs
```

---

## Configs (Hydra)

The default composition in `configs/config.yaml` selects `data/default`,
`model/mlp`, `trainer/local`. Override anything from the CLI:

```bash
make train OVERRIDES="trainer=cloud model.hidden_dim=256 seed=7"
make train OVERRIDES="experiment=example"          # composes a named experiment
```

Config schemas are typed dataclasses in `src/ml_template/config_schemas.py`,
registered with Hydra's `ConfigStore`. Type errors and missing fields fail at
load time, not three minutes into a run.

## Tracking (MLflow)

Local default writes to `./mlruns/`. To point at a remote tracking server:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.internal:5000
make train
```

The `MLflowConfig` in code reads `tracking_uri` from the resolved config,
which you can override per run.

---

## Project layout

```
src/ml_template/      package — all production code lives here
  config_schemas.py   typed Hydra structured configs
  data/               datasets and dataloaders
  models/             nn.Modules
  training/           train/eval loop
  utils/              device / seed / logging helpers
  cli/                Hydra entrypoints (train, eval)
configs/              Hydra config tree (data/, model/, trainer/, experiment/)
scripts/              ad-hoc utilities (smoke test, device check)
tests/                pytest suite — mirrors src/
docs/                 research and project documentation
notebooks/            exploratory notebooks (excluded from strict lint/types)
data/                 gitignored — datasets / artifacts
outputs/              gitignored — Hydra runtime outputs
mlruns/               gitignored — MLflow file-store runs
```

## Documentation

Research and project docs live under [`docs/`](docs/README.md). Start with
`docs/research_statement.md` for the project's goal and `docs/methods.md` for
the current architecture and training procedure.

## For coding agents

This repo is laid out for agent-driven development. Read
[`CLAUDE.md`](CLAUDE.md) before making changes — it tells agents where things
live, what verification to run, and what not to touch without asking.
