# EXP-0001 — <slug>

> Template. Copy to `EXP-NNNN-<slug>.md` for each significant experiment / run
> group. Numbered for stable references from PRs and other docs.
>
> **Never invent metrics. Only record runs that actually executed.**

## Motivation

What question does this experiment answer? Link the relevant section of
`research_statement.md` if applicable.

## Hypothesis

The specific prediction being tested.

## Setup

- **Branch / commit:** `<sha>`
- **Config:** `<path/to/config.yaml>` plus CLI overrides:
  ```
  make train OVERRIDES="trainer=cloud model.hidden_dim=256"
  ```
- **Hardware:** (e.g., 1× A100 80GB on RunPod)
- **MLflow experiment:** `<experiment_name>`
- **MLflow run id(s):** `<run_id>`

## Results

| Metric | Value |
|---|---|
| val/loss | 0.000 |
| val/accuracy | 0.000 |

(Plot links into `docs/figures/`.)

## Takeaways

- ...

## Follow-ups

- ...
