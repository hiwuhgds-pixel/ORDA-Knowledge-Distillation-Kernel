# orda_ce_kernel Test And Benchmark Suite

This suite is organized for installed-package testing with the `src/` layout.
From a fresh environment, install the package first:

```
python -m pip install -e ".[test]"
```

## Target Hardware

Orda is written for **Tesla T4 / fp16**.

**Do not generalise Orda benchmark numbers to A100/H100** — the design trade-offs here are currently only tested on Tesla T4. Re-run on the target hardware before claiming.

CUDA/Triton correctness tests and all benchmark scripts require a real Linux CUDA/HIP environment where `orda_ce_kernel.is_available()` returns `True`. When that is not true, GPU tests skip and benchmark scripts exit without numbers.

## One-click runner (Colab / T4)

```
python scripts/run_all_test_colab.py
```

The one-click runner writes artifacts under `--output-dir` (default `benchmark_results/`) using this layout:

```
benchmark_results/
    json/    # one JSON per benchmark step
    csv/     # one CSV per benchmark step
    logs/    # one .log per step + run_all.log master log
```

Useful flags:

- `--skip-large` — skip the `bench_end_to_end --mode orda-large` config (16x1024) when free VRAM is tight.
- `--skip-correctness` / `--skip-unit` — drop the pytest gates.
- `--dry-run` — print the planned command list without executing.

## Layout

- `tests/unit/`: CPU-safe tests for the public API, teacher objects, `DistillationLoss`, PyTorch fallback, pre-dispatch validation, resolver/dispatcher behavior, OOM retry dispatch logic, quant/dequant helpers, timing statistics, benchmark artifacts, and runtime option flow.
- `tests/correctness/`: CUDA/Triton numerical tests against PyTorch/FP64 references. These cover CE-only, CE+KL, standalone KL, tied/separate/precomputed KD modes, default KD weights, the raw loss component contract, upstream autograd scaling, supported reduction modes, non-power-of-two vocab sizes, dtype coverage, chunking, `max_fused_size`, label smoothing, `ignore_index`, extreme logits, finite gradients, and deterministic outputs.
- `tests/benchmarks/`: benchmark entry points for CE-only, CE+KL, KL accuracy, KL throughput, estimated memory bandwidth, end-to-end training-like paths, and an Nsight wrapper.
- `tests/utils/`: shared runtime guards, FP64 references, comparison metrics, CUDA event timing, OOM handling, and output helpers.
- `tests/VALIDATION_EN.md` / `tests/VALIDATION_VI.md`: verification checklists and validation gates before releasing numbers.
- `scripts/check_ast_imports.py`: cross-platform AST + import-hygiene gate.
- `scripts/run_all_test_colab.py`: master runner script for T4/Colab.

## Install and Static Checks

```
python -m pip install -e ".[test]"
python -m pytest tests/unit -q
python scripts/check_ast_imports.py
```

## CUDA Correctness

Run on a GPU machine only:

```
python -m pytest tests/correctness -q
```

Expected coverage:

- CE-only and CE+KL losses/gradients match PyTorch/FP64 references.
- `student_ce`, `teacher_ce`, and `kl` are reported components; total loss must apply `student_ce_weight`, `teacher_ce_weight`, and `kd_weight` explicitly.
- Standalone KL kernel matches PyTorch for multiple temperatures, online/fixed softmax, and fast-math flag combinations.
- `teacher_ce_weight=None` defaults to pure KD (`0.0`) for `SeparateTeacher` and `PrecomputedTeacher`, while preserving tied CE (`1.0`) for `TiedTeacher`.
- `DistillationLoss` must match the functional API, and backward must respect upstream gradient scaling.
- Hidden-state gradients for ignored tokens are zero.
- Supported reductions `mean` and `sum` are checked.
- Shapes include small batches, large token counts, and non-power-of-two vocab sizes.
- Dtypes include fp16, fp32, and bf16 when the GPU reports bf16 support.
- Extreme-logit cases must keep loss and gradients finite.

## Benchmarks

Benchmarks are not CPU fallback scripts. If CUDA/Triton is unavailable, they print a skip message and exit without producing numbers.

```
python -m tests.benchmarks.bench_ce_only
python -m tests.benchmarks.bench_ce_kl
python -m tests.benchmarks.bench_kl_accuracy
python -m tests.benchmarks.bench_kl_throughput
python -m tests.benchmarks.bench_memory_bandwidth --verify
python -m tests.benchmarks.bench_end_to_end
```

Smoke examples:

```
python -m tests.benchmarks.bench_ce_only --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2
python -m tests.benchmarks.bench_ce_kl --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2
python -m tests.benchmarks.bench_memory_bandwidth --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2 --verify
```

### Script CLI Parameter Documentation

Besides the common CLI arguments (`--configs`, `--vocab-size`, `--hidden-dim`, `--dtype`, `--warmup`, `--steps`, `--repeats`, `--seed`), several benchmark scripts accept specific parameters or have overrides:

1. **`bench_kl_accuracy.py`**:
   * `--sample-frac` (default: `0.25`): Sample fraction of validation dataset to evaluate.
   * *Override*: `--hidden-dim` defaults to `1024`.
2. **`bench_kl_throughput.py`**:
   * `--student-layers` (default: `4`): Number of student layers for simulation.
   * `--teacher-layers` (default: `12`): Number of teacher layers for simulation.
   * `--grad-accum` (default: `4`): Gradient accumulation steps.
   * `--lambda-student` (default: `1.0`): Coefficient for student loss.
   * `--no-compile` (flag): Disables `torch.compile` on model definitions.
   * *Override*: `--hidden-dim` defaults to `1024`; `--warmup` defaults to `2`.
3. **`bench_end_to_end.py`**:
   * `--layers` (default: `2`): Simulated transformer layers.
   * `--heads` (default: `8`): Simulated transformer attention heads.
   * `--batch-size` (default: `None`): Explicit batch size to override default.
   * `--seq-len` (default: `None`): Explicit sequence length to override default.
   * `--no-compile` (flag): Disables model compilation.
   * *Override*: `--hidden-dim` defaults to `2048`; `--steps` defaults to `15`.
4. **`profile_wrapper.py`**:
   * `--target-module` (default: `"tests.benchmarks.bench_ce_only"`): Benchmark target module.
   * `--target-args` (default: `""`): Target script arguments.
   * `--ncu` (default: `None`): Path to Nvidia NCU executable.
   * `--nsys` (default: `None`): Path to Nvidia Nsys executable.
   * `--kernel` (default: `None`): Target kernel name filter.
   * *Override*: `--output-dir` defaults to `profile_results`.

## Benchmark Outputs

Each benchmark accepts:

```
--output-json benchmark_results/name.json
--output-csv benchmark_results/name.csv
```

When benchmarks are run directly, JSON/CSV are written exactly to the paths passed on the CLI. Those direct runs do not create a `logs/` directory unless the caller redirects stdout/stderr separately.

JSON contains `metadata` (benchmark name, timestamp, PyTorch/CUDA version, device info, CLI args) and `rows` (one per config/method with latency, peak VRAM, status).

### Timing statistics columns

Every speed bench runs `--repeats` outer loops × `--steps` timed iterations:
- Default parameters: `--repeats 5 --steps 20` (100 samples total), except for `bench_end_to_end` which uses `--steps 15` (75 samples total).
- Default warmup: `--warmup 5`, except for `bench_kl_throughput` which uses `--warmup 2`.

Metrics:
- `latency_ms` — trimmed mean (10%) across all samples.
- `latency_ms_std`, `latency_ms_min`, `latency_ms_max` — raw spread.
- `latency_ms_p50`, `latency_ms_p95` — robust central + tail latency.
- `cv_pct` — coefficient of variation. If `cv_pct > 15%` the script prints `[WARN]` — don't use that row for headline claims.

## Profiling

```
python -m tests.benchmarks.profile_wrapper --dry-run
python -m tests.benchmarks.profile_wrapper --mode all --target-module tests.benchmarks.bench_ce_kl
```

## Valid Skips

Skips are valid when:

- CUDA/HIP is not available.
- Triton kernels are not importable or `orda_ce_kernel.is_available()` is false.
- bf16 is requested on a GPU that does not support bf16. Correctness tests skip bf16 gracefully; benchmark runner scripts raise a RuntimeError if requested on unsupported devices.

## Checklist Before Real Numbers

1. Install with `python -m pip install -e ".[test]"`.
2. Run `python scripts/check_ast_imports.py`.
3. Run `python -m pytest tests/unit -q`.
4. On GPU, run `python -m pytest tests/correctness -q`.
5. Run benchmark smoke configs with JSON/CSV output.
6. Run full benchmark configs only after the correctness gate passes.
