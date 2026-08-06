# Test suites

Run commands from the repository root. Install the test extra first:

```bash
pip install '.[test]'
```

The CPU and GPU runners own separate suites. The CPU runner never collects CUDA tests; the GPU runner never collects CPU reference tests.

## CPU runner

```bash
python test/run_test_cpu.py
```

This runs, in one pytest invocation:

| Suite | Path |
|---|---|
| Unit contracts and implementation tests | `test/unit` |
| CPU Torch-reference correctness | `test/correctness/test_torch_backend_cpu.py` |
| Autotune multiprocess behavior | `test/nightly/test_autotune_multiprocess.py` |

All additional arguments are forwarded to pytest:

```bash
python test/run_test_cpu.py -k chunking -x
python test/run_test_cpu.py --collect-only
```

The runner writes the combined UTF-8 terminal output to `logs/cpu/run_test_cpu.log`. A numeric suffix prevents overwrites.

## GPU runner

```bash
python test/run_test_gpu.py
```

With no selection flags, it presents an interactive menu. Select an item by number, a comma-separated list or range (`1,3-5`), its name, or `all`. The four Cartesian routes are direct options 6-9; selecting `6-9` does not remove or sample any Cartesian cases.

| Name | Suite |
|---|---|
| `dlogits` | `test/correctness/test_dlogits_contract_cuda.py` |
| `kl-variants` | `test/correctness/test_kl_variants_cuda.py` |
| `low-level` | `test/correctness/test_low_level_ops_cuda.py` |
| `public-api` | `test/correctness/test_public_api_cuda.py` |
| `autotune` | `test/nightly/test_autotune_full_batch_cuda.py` |
| `cartesian-tied` | `test/correctness/test_cartesian_cuda.py` |
| `cartesian-separate` | `test/correctness/test_cartesian_cuda.py` |
| `cartesian-precomputed_logits` | `test/correctness/test_cartesian_cuda.py` |
| `cartesian-precomputed_hidden_weight` | `test/correctness/test_cartesian_cuda.py` |

Non-interactive examples:

```bash
# One or several direct GPU suites.
python test/run_test_gpu.py --test public-api
python test/run_test_gpu.py --test dlogits --test kl-variants

# Every GPU suite. Cartesian runs once per route.
python test/run_test_gpu.py --all

# Cartesian routes are direct options 6-9.
python test/run_test_gpu.py --test cartesian-tied
python test/run_test_gpu.py --route tied
```

The GPU runner writes the combined UTF-8 terminal output to `logs/<gpu>/test/run_test_gpu.log`. A numeric suffix prevents overwrites.

The Cartesian routes partition the complete matrix by teacher representation: `tied`, `separate`, `precomputed_logits`, and `precomputed_hidden_weight`. Each route remains grouped internally to bound memory use and preserve failure aggregation.

## Cartesian filters and reruns

Filters apply only to `cartesian-*` options and can be repeated to choose multiple values. Use `--list` to show the available test groups, routes, and GPU-supported dtype total before running anything.

```bash
python test/run_test_gpu.py --list
python test/run_test_gpu.py --route separate --kl forward --dtype float32
python test/run_test_gpu.py --test cartesian-tied --labels all_ignored
python test/run_test_gpu.py --ordinal 12345
```

Supported filters are `--kl`, `--dtype`, `--api`, `--chunk`, `--labels`, `--outcome`, and `--ordinal`. A requested `bfloat16` case is skipped when the active GPU lacks native BF16 support.

## Direct pytest

The runners are convenience routers; direct pytest commands remain supported:

```bash
python -m pytest test/unit -q
python -m pytest test/correctness/test_public_api_cuda.py -q
python -m pytest test/correctness/test_cartesian_cuda.py -q
```

To pass pytest options through the GPU runner, place them after `--`:

```bash
python test/run_test_gpu.py --test public-api -- -k current_device_differs -q
```

For CUDA/HIP environment setup and the full manual validation gate, see [`docs/gpu-validation.md`](../docs/gpu-validation.md).
