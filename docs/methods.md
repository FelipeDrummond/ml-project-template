# Methods

> Single source of truth for the **current** architecture, loss, and training
> procedure. Keep in sync with code. References to code paths in inline backticks
> are validated by a pre-commit hook (`scripts/check_docs_methods_paths.py`):
> if a path is renamed or removed, the commit fails.

## Overview

(One paragraph: what does this system do, end to end?)

## Data

- **Source / generation:** see `docs/data_card.md`.
- **Loader:** `src/ml_template/data/dataset.py`.
- **Splits:** train / val constructed in `build_dataloaders`; deterministic
  given `seed`.

## Model

- **Architecture:** MLP placeholder — `src/ml_template/models/mlp.py`.
- **Hyperparameters:** see `configs/model/mlp.yaml`. Defaults documented in
  `src/ml_template/config_schemas.py` (`ModelConfig`).

## Training

- **Loop:** `src/ml_template/training/loop.py`.
- **Loss:** cross-entropy (placeholder; replace per project).
- **Optimizer:** AdamW (`lr`, `weight_decay` in `TrainerConfig`).
- **Device:** resolved by `src/ml_template/utils/device.py`. Local profile uses
  `auto` (CUDA → MPS → CPU); cloud profile pins `cuda` and fails loud if
  unavailable.
- **Reproducibility:** `src/ml_template/utils/seed.py` seeds python / numpy /
  torch (cpu+cuda+mps) and `PYTHONHASHSEED`.

## Evaluation

- **Loop:** `_evaluate` inside `src/ml_template/training/loop.py` (model.eval +
  inference_mode).
- **Metrics:** mean cross-entropy loss, top-1 accuracy.

## Tracking

- **Backend:** MLflow. URI from `MLflowConfig.tracking_uri` (default
  `file:./mlruns`; override via `MLFLOW_TRACKING_URI`).
- **What is logged:** flattened config as params; per-step `train/loss`;
  per-epoch `val/loss` and `val/accuracy`.

## Smoke test (canonical sanity check)

`scripts/overfit_one_batch.py` — must drive loss → ~0 in <200 steps. Run after
any change to the training path.
