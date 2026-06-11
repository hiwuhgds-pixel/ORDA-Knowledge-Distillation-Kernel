# Validation Gates

Use these gates before using benchmark numbers in a report.

## Gate 0: Scope And Import Hygiene

Run on any machine from the repository root. The check script is plain
Python so the same command works on Linux, macOS, and Windows:

```
python scripts/check_ast_imports.py
```

Expected result:

- AST parsing succeeds for every Python file under `tests/`.
- The import-hygiene check finds no `sys.path` / `sitecustomize` /
  `PYTHONPATH` usage in test code.
- No `src/**` file was changed for test convenience.

## Gate 1: CPU-Safe Unit Tests

```
python -m pytest tests/unit -q
```

Expected result: all unit tests pass. This gate verifies:

- public API exports, `DistillationLoss`, teacher objects, and validation errors before CUDA dispatch;
- runtime option flow through public API arguments and `KernelConfig`;
- core runtime modules receive execution options through function arguments;
- resolver chunk sizing and max-chunk limits;
- dynamic dispatcher OOM retry/cache behavior using mocks;
- quant/dequant shape, zero-row, stochastic seed, and target-row behavior;
- timing statistics calculations (trimmed mean, stdev, percentile, CV% logic);
- benchmark utility parsing, artifact writing, guarded compile fallback, and
  dtype aliases.

## Gate 2: CUDA Correctness

Run only on a Linux CUDA/HIP machine where `orda_ce_kernel.is_available()` is
`True`:

```
python -m pytest tests/correctness -q
```

Expected result: all GPU correctness tests pass. Coverage includes:

- fused CE+KL loss and gradients against FP64 reference;
- CE-only `mean` and `sum` reductions;
- explicit and dynamic chunking;
- `ignore_index` hidden-token gradient masking;
- label smoothing and `student_ce_weight`;
- the contract that `student_ce`, `teacher_ce`, and `kl` are reported components, with total loss applying `student_ce_weight`, `teacher_ce_weight`, and `kd_weight`;
- `teacher_ce_weight=None` defaults to pure KD for `SeparateTeacher`/`PrecomputedTeacher` while preserving teacher CE for `TiedTeacher`;
- `DistillationLoss` matches the functional API and backward respects upstream gradient scaling;
- non-power-of-two vocabulary sizes;
- fp16, fp32, and supported bf16 inputs;
- standalone Triton KL loss/gradient;
- KL temperature, online softmax, fast-math, and multiply-not-divide flags;
- all-ignored KL rows;
- numerical stress for extreme logits;
- tied weight (shared embedding and head weight) student-teacher gradients verification;
- deterministic output for identical inputs.

## Gate 3: Benchmark Smoke

Run only after Gate 2 passes:

```
python -m tests.benchmarks.bench_ce_only --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2 --output-json benchmark_results/ce_only_smoke.json --output-csv benchmark_results/ce_only_smoke.csv
python -m tests.benchmarks.bench_ce_kl --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2 --output-json benchmark_results/ce_kl_smoke.json --output-csv benchmark_results/ce_kl_smoke.csv
python -m tests.benchmarks.bench_memory_bandwidth --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2 --verify --output-json benchmark_results/memory_smoke.json --output-csv benchmark_results/memory_smoke.csv
```

Expected result:

- scripts print real CUDA timings or explicit OOM rows;
- JSON/CSV files are created when requested, including `status=skipped` rows
  when the package, CUDA, or Triton kernels are unavailable;
- metadata records the expected GPU, CUDA/HIP/PyTorch versions, and CLI args;
- no script emits CPU fallback numbers.
- `bench_memory_bandwidth --verify` runs finite CE/CE+KL loss and gradient
  sanity checks against PyTorch before reporting relative traffic estimates.

## Gate 4: Full Benchmark Collection

Run full configs only after benchmark smoke passes:

```
python -m tests.benchmarks.bench_ce_only --output-json benchmark_results/ce_only.json --output-csv benchmark_results/ce_only.csv
python -m tests.benchmarks.bench_ce_kl --output-json benchmark_results/ce_kl.json --output-csv benchmark_results/ce_kl.csv
python -m tests.benchmarks.bench_kl_accuracy --output-json benchmark_results/kl_accuracy.json --output-csv benchmark_results/kl_accuracy.csv
python -m tests.benchmarks.bench_kl_throughput --output-json benchmark_results/kl_throughput.json --output-csv benchmark_results/kl_throughput.csv
python -m tests.benchmarks.bench_memory_bandwidth --verify --output-json benchmark_results/memory_bandwidth.json --output-csv benchmark_results/memory_bandwidth.csv
python -m tests.benchmarks.bench_end_to_end --output-json benchmark_results/end_to_end.json --output-csv benchmark_results/end_to_end.csv
```

Interpretation rules:

- `bench_kl_accuracy` includes `fp32_full`, `fp16_full`,
  `fp16_f32_row_sum`, `sample_25`, and `triton` rows.
- `bench_kl_throughput` includes `no_kl`, `kl_triton`,
  `fp16_f32_row_sum`, `kl_full`, `chunked_pytorch_kl`, and
  `kl_sample_25` rows.
- `status=ok` rows are usable timing rows.
- `status=oom` rows are valid capacity observations, not latency data.
- `NaN` latency must not be averaged into performance summaries.
- Estimated GiB/s is a relative estimate. Use Nsight for authoritative memory
  metrics.
- Timed rows produced through `cuda_benchmark` (i.e. `status=ok` and
  `status=oom` rows from a variant that actually ran) carry `latency_ms_std`,
  `_min`, `_max`, `_p50`, `_p95`, `cv_pct`, and `repeats`. `status=skipped`
  rows and OOM rows that short-circuited before timing started do not — their
  stats fields are absent on purpose so downstream tooling does not confuse
  a never-measured variant with a measured one. Rows where `cv_pct > 15%`
  are flagged with `[WARN]` during the run; they must be re-run or excluded
  from headline claims.
- `bench_kl_throughput` rows use `vs_no_kl` and
  `peak_vram_delta_vs_no_kl_mib` for comparator fields. Artifacts that contain
  `vs_kl_triton` or `peak_vram_delta_vs_kl_triton_mib` were produced by an
  earlier script revision and should be regenerated before reporting.
- Benchmark rows may include `loss_compile`, indicating whether PyTorch
  baseline losses and eligible student/teacher models attempted guarded
  `torch.compile` with eager fallback.

## Gate 5: Profiling

For command validation:

```
python -m tests.benchmarks.profile_wrapper --dry-run
```

For real profiling:

```
python -m tests.benchmarks.profile_wrapper --mode all --target-module tests.benchmarks.bench_ce_kl
```

Expected result: Nsight commands are printed in dry-run mode, or `.ncu-rep` /
`.nsys-rep` artifacts are generated when tools and CUDA are available.

## Final Sign-Off

Before reporting numbers, record:

- exact commands run;
- unit/correctness pass results;
- GPU model and software metadata from JSON;
- benchmark configs, dtype, warmup, steps, and seed;
- OOM rows and skipped benchmarks;
- whether Nsight profiling was run or only dry-run command generation was done;
- the hardware caveat (T4 fp16; numbers do not generalise to Hopper/Ada).
