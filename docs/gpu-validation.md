# GPU validation

Run these gates with a CUDA or ROCm PyTorch build and Triton installed. Start from a clean process so JIT and autotune state are representative.

## Core correctness

```bash
python -m pytest test/correctness/test_dlogits_contract_cuda.py -q
python -m pytest test/correctness/test_public_api_cuda.py -q
python -m pytest test/correctness/test_low_level_ops_cuda.py -q
```

These suites compare every teacher path and KL family against independent Torch or fp64 references, including inactive branches, ignored non-finite rows under strict and trusted-label modes, noncontiguous inputs, and precomputed-logit dtype conversion. `test_public_api_cuda.py` also compares the aligned-span Triton path against its torch reference for both binary objectives, CE enabled/disabled, and one versus multiple chunks; it checks loss, hidden gradients, and weight gradients. `test_dlogits_contract_cuda.py` pins the raw ignored-row zero contract used by the trusted weight-gradient accumulation path across every teacher mode and the aligned-span kernel.

The aligned-span cases are the hardware gate for both canonical stages:

- `ALIGNED_STATS`: ignored-row-safe NLL and saved LSE
- `ALIGNED_GRAD`: `a_i * (p - onehot)` written back to the logits buffer
- one-chunk logits reuse and multi-chunk recomputation
- FP64 span prefix/difference math around FP32 kernel statistics

## Support-sentinel and block-independence gates

```bash
python -m pytest test/correctness/test_kl_variants_cuda.py -q
```

Beyond kernel/reference parity, this suite pins the numerically sensitive support cases:

- Rényi and Rényi-reverse over teacher columns with `-inf` support, checked against the fp64 reference at partial base support
- `reverse`, `jeffreys`, and `entropy_adaptive` return `+inf` loss and zeroed gradients on missing support across `max_fused_size` in `{32, 128, 512}`, so the sentinel decision cannot depend on the running per-block normalization
- `alpha_divergence` at `alpha` near `0` and `1` matches the reference value and stays non-negative
- frozen tied and separate teachers leave `teacher_hidden`/`teacher_weight` gradients `None` while the student path still receives gradient

## Full Cartesian matrix

The GPU runner owns the direct correctness files, the full-batch autotune gate, and Cartesian. Select one or more exact suites with repeatable `--test`:

```bash
python test/run_test_gpu.py --list
python test/run_test_gpu.py --test public-api
python test/run_test_gpu.py --test low-level --test kl-variants
python test/run_test_gpu.py --test autotune
python test/run_test_gpu.py --all
```

The Cartesian suite includes `student_ce_weight=0`, strict label failures, trusted layouts, all chunk policies, all public/direct APIs, and supported dtypes. Cases execute as grouped pytest items keyed by `(teacher_representation, kl_variant, dtype, api_level)`. The runner has four teacher-representation routes; KL variant is an independent filter. With no selection flags, it prints the same numbered interactive selection style as the benchmark routers. The four Cartesian routes are direct options 6-9:

```bash
python test/run_test_gpu.py
python test/run_test_gpu.py --test cartesian-tied
python test/run_test_gpu.py --route tied --kl forward --dtype float32
```

Selecting options 6-9 runs all four routes in separate sequential pytest subprocesses instead of materializing the 2,073,600-case matrix in one process. Cartesian always runs with autotune disabled; the dedicated autotune suites own that coverage. Filters also include repeatable `--kl`, plus `--api`, `--chunk`, `--labels`, `--outcome`, and repeatable `--ordinal`.

Shard any selected route across workers or machines:

```bash
export ORDA_CARTESIAN_SHARD_COUNT="8"
export ORDA_CARTESIAN_SHARD_INDEX="0"  # 0..7
python test/run_test_gpu.py --route tied --kl forward -q
```

Run every shard index with the same shard count.

Rerun specific cases by ordinal, without shard variables:

```bash
python test/run_test_gpu.py --ordinal 123456 --ordinal 123457 -q
```

## Autotune and large blocks

```bash
python test/run_test_gpu.py --test autotune -q
```

This gate runs the supported-path matrix, verifies the bounded full-plan timing contract, compares explicit 16K/32K launches against Torch, and asserts that the 32K budget exposes both large block sizes. `autotune=True` and `"Max"` use that same candidate set; `"Max"` performs the deeper robust measurement.

GPU suites are not scheduled in this repository's CI. Run the relevant commands manually on each CUDA or ROCm architecture you need to support, including the ROCm launch-candidate coverage when applicable. Record the hardware and runtime details with the result.

## Multi-device guard

On a machine with at least two visible GPUs:

```bash
python test/run_test_gpu.py --test public-api -- \
  -k "current_device_differs" -q
```

The test keeps the ambient device on `cuda:0`, executes tensors on `cuda:1`, and checks that the launch guard restores the ambient device.

## Final GPU gate

```bash
python test/run_test_gpu.py --all -q
```

The final gate runs each direct suite and autotune suite once, then executes Cartesian as four sequential teacher-representation routes. Record GPU model, Torch version, Triton distribution/version, CUDA or HIP runtime, and driver version with the result.
