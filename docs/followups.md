# Follow-ups

Known gaps in the template that aren't worth blocking on but should be
addressed eventually. Priority-ordered (highest first).

---

## 1. Verify everything on real CUDA

**Status**: never run. Mac/CPU only.

**What's at risk**: every CUDA-only code path. Specifically:
- `trainer.precision="bf16"` + Accelerate autocast
- `trainer.precision="fp16"` + Accelerate's GradScaler integration
- `torch.optim.AdamW(..., fused=True)` (only activates on CUDA)
- `trainer.compile_mode` modes ("default" / "reduce-overhead" / "max-autotune")
- `pynvml` util% logging
- TF32 actually doing anything (silent no-op on non-Ampere)

**How to verify**: rent a 1-hour 3090/4090/A100 (RunPod, Lambda, Vast).
On the box:
```bash
git clone <this-repo>
cd ml-project-template
make install-cuda
make dev-run                                    # CUDA-aware fast_dev_run
make train OVERRIDES="trainer=cloud trainer.epochs=2 data.n_samples=1024"
```

**Acceptance**:
- `make install-cuda` resolves cleanly
- bf16 autocast trains without NaN
- `gpu/util_pct` and `gpu/mem_*` metrics appear in MLflow
- `torch.compile` (set `compile_mode="default"`) works after warm-up
- `nvidia-smi` confirms TF32 enabled (~30% matmul speedup vs no TF32)

---

## 2. Verify `uv sync --extra dev --extra cuda` on Linux

**Status**: only `--extra dev --extra cpu` exercised, on macOS arm64.

**What's at risk**: the `[tool.uv.sources]` + `[[tool.uv.index]]` block in
`pyproject.toml`. Specifically that `marker = "sys_platform == 'linux'"`
correctly routes to `download.pytorch.org/whl/cu121` on Linux while
keeping macOS on default PyPI.

**How to verify**: same rental box as #1. Compare:
```bash
uv sync --extra dev --extra cuda
uv pip show torch | grep -i location  # confirm the CUDA wheel path
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

**Acceptance**: `torch.cuda.is_available() == True`, `torch.version.cuda`
reports 12.1, no PyPI fallback for torch.

Bundles naturally with #1.

---

## 3. Verify spot-upload against real S3 or GCS

**Status**: tested only via `file://` URIs. fsspec abstracts the same
interface, but real cloud storage has edge cases the local filesystem
doesn't exercise:
- Authentication failures
- Mid-upload network errors
- S3 rename = copy + delete (not atomic, unlike POSIX)
- GCS eventual consistency on directory listings

**How to verify**:
1. Create a throwaway bucket: `aws s3 mb s3://ml-template-spot-test-<random>`
2. Install the backend: `uv sync --extra dev --extra cuda --extra s3`
3. Run with the URI:
   ```bash
   export CHECKPOINT_URI=s3://ml-template-spot-test-<random>/run-1
   make train OVERRIDES="trainer=cloud trainer.checkpoint_uri=$CHECKPOINT_URI trainer.epochs=2"
   ```
4. Manually trigger SIGTERM mid-run: `kill -TERM <pid>` (from another shell)
5. Verify the SIGTERM checkpoint and at least one periodic checkpoint
   exist in S3.
6. Resume from the remote URI in a fresh run:
   ```bash
   make train OVERRIDES="trainer=cloud trainer.resume_from=$CHECKPOINT_URI/sigterm_ckpt"
   ```

**Acceptance**: SIGTERM upload completes within 30s; resume from `s3://`
works without code changes.

Repeat for `gs://` if you'll use GCP.

---

## 4. Add local tests for the SIGTERM-handler code path

**Status**: `tests/test_spot.py` covers `upload_checkpoint`,
`maybe_resolve_remote_resume`, and end-to-end *periodic* upload + remote
resume. The actual `install_sigterm_handler` context manager is never
exercised — every line inside the signal handler is untested in CI.

**What's at risk**: the spot-survival promise. If the handler regresses
(wrong exit code, missed upload, swallowed exception bubbles up,
previous-handler not restored), nothing catches it before a real spot
reclaim does. Item #3 covers this manually on a cloud box, but a
regression should fail locally before it ships.

**Tests to add** (all use `file://` URIs and
`os.kill(os.getpid(), signal.SIGTERM)` — no cloud needed):

1. **`test_sigterm_handler_saves_local_checkpoint_no_uri`** — install the
   handler with a stub `take_checkpoint` returning a known meta and
   `checkpoint_uri=None`. `os.kill` self with SIGTERM. Assert:
   `SystemExit` raised with code 143; `<output_root>/sigterm_ckpt/checkpoint_meta.json`
   exists; meta round-trips correctly.

2. **`test_sigterm_handler_uploads_to_file_uri`** — same setup with
   `checkpoint_uri=file://<tmp>/remote`. Assert: remote dir exists with
   `checkpoint_meta.json`, local `sigterm_ckpt` exists, `SystemExit(143)`.

3. **`test_sigterm_handler_survives_failed_upload`** — pass a URI that
   `fsspec` will reject (e.g. an unregistered scheme `bogus://x/y` or a
   `file://` pointing at a path the test makes read-only). Assert: local
   `sigterm_ckpt` still on disk; the upload exception was logged via
   `caplog` but not re-raised; `SystemExit(143)` still propagates.

4. **`test_sigterm_handler_restores_previous_handler`** — set a custom
   handler with `signal.signal(SIGTERM, custom)` before entering the
   `with`, exit the block normally (no SIGTERM), assert
   `signal.getsignal(SIGTERM) is custom` after. Pins the `try/finally`
   contract in `install_sigterm_handler`.

5. **`test_sigterm_during_training_then_resume`** — interrupt a real
   `train()` mid-run by monkeypatching a hook the loop calls per step
   (e.g. `mlflow.log_metric`) to call `os.kill(os.getpid(), SIGTERM)`
   after the Nth invocation. Catch `SystemExit`, assert
   `<output_root>/sigterm_ckpt` exists, then start a fresh `train()`
   with `resume_from=<output_root>/sigterm_ckpt` and assert it advances
   `start_epoch` and produces final metrics. This is the only test that
   exercises the *full* reclaim cycle (handler fires → checkpoint
   written → next process resumes from it).

**Acceptance**: all five new tests pass on Mac/CPU; they live in
`tests/test_spot.py` next to the existing ones; none require cloud
credentials. With these in place, item #3 narrows to "verify the cloud
backends behave like `file://`," not "verify the handler is correct."

**Stretch** (nice-to-have, may be flaky): an equivalence test that
runs N epochs uninterrupted, then K epochs + SIGTERM-resume + (N-K)
epochs, and asserts the two final `val/loss` values match within
tolerance on CPU + fixed seed. Catches RNG-state-not-preserved
regressions in `Accelerator.save_state`. Skip if it proves
non-deterministic in practice — Accelerate's RNG capture is supposed
to make this work but the equivalence is empirical, not guaranteed.

---

## 5. Add a CLAUDE.md path-table validator

**Status**: missing. The `Where things live` table in `CLAUDE.md`
references many paths (`src/ml_template/training/loop.py`,
`tests/test_train_smoke.py`, etc.). Today, none of them are validated —
a refactor that moves a file silently invalidates the table.

The pattern already exists: `scripts/check_docs_methods_paths.py`
validates `docs/methods.md`. Generalize it to also check `CLAUDE.md`.

**How to implement** (~15 lines):
1. Refactor `scripts/check_docs_methods_paths.py` to take the doc path
   as an argument, OR copy the file to `scripts/check_claude_md_paths.py`.
2. Add a hook entry in `.pre-commit-config.yaml`:
   ```yaml
   - id: claude-md-paths
     name: CLAUDE.md references valid code paths
     entry: python scripts/check_claude_md_paths.py
     language: system
     files: ^CLAUDE\.md$
     pass_filenames: false
   ```
3. (Optional) AGENTS.md is currently a copy of CLAUDE.md — same
   validation should apply.

**Acceptance**: a deliberately broken path in CLAUDE.md (e.g. rename
`src/ml_template/training/loop.py` → `whatever.py` in the table)
causes `pre-commit run --all-files` to fail.

---

## 6. Verify CI's `fast_dev_run` step

**Status**: added in commit `d18a2e1` to `.github/workflows/ci.yml`,
never run because no PR has triggered it.

**What's at risk**: trivial things — a typo in the step name, a
missing dep at the GHA level, the runner not having git available
(unlikely). Probably works.

**How to verify**: open any throwaway PR (e.g. tiny README typo fix)
and watch the Actions tab. The step should run after pytest and
print the round-trip success line.

**Acceptance**: green CI on a PR with the `fast_dev_run end-to-end smoke`
step visible in the run log.

---

## Deliberately not on this list

These are defensible to add per-project but don't belong in the template:
- 8-bit AdamW (`bitsandbytes`) — only matters when memory-bound
- MFU computation via `torch.utils.flop_counter` — chose util% for simplicity
- Optuna multirun sweeper — manual `OVERRIDES="..."` sweeps work fine at personal scale
- Multi-GPU (DDP/FSDP/ZeRO) — single-rented-GPU is the assumed shape
- Activation checkpointing — only matters for huge models
- Dataset preprocessing cache — domain-specific
- OOM auto-retry / batch-size auto-tune — brittle, surfaces bugs as perf regressions

If your project's needs grow into one of these, lift it from a real
project rather than pre-baking it here.
