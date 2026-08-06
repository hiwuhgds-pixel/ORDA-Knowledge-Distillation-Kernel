# API reference

Complete public contract: every export, parameter, constraint, and error. For task-oriented usage, see [guide.md](guide.md). For routing to kernel paths and variant support, see [variants.md](variants.md).

Public surface (`import orda_kernel`):

| Group | Exports |
|---|---|
| Tokenwise API | `distillation_loss`, `DistillationLoss`, `DistillationLossOutput` |
| Teacher descriptors | `TiedTeacher`, `SeparateTeacher`, `PrecomputedTeacher` |
| Configuration | `KernelConfig`, `ChunkSize`, `EntropyGate` |
| Aligned-span API | `aligned_span_distillation_loss`, `AlignedSpanPairs`, `AlignedLikelihoodConfig`, `CrossVocabTeacher`, `PrecomputedSpanTeacher`, `build_precomputed_span_teacher` |
| Precomputed-input builders | `build_exact_token_map`, `project_teacher_logits`, `mix_teacher_logits`, `generalized_jensen_shannon` |
| Runtime controls | `is_available`, `reset_autotune_cache`, `begin_autotune_run`, `__version__` |

Low-level ops (`orda_kernel.ops.*`) are documented in [section 8](#8-low-level-ops).

---

## 1. `distillation_loss`

```python
distillation_loss(
    student_hidden, weight, labels, teacher, *,
    student_ce_weight=1.0, teacher_ce_weight=None,
    kl_weight=0.0, kl_temperature=1.0, kl_teacher_temperature=None,
    kl_divergence="forward", kl_alpha=None, kl_beta=None, entropy_gate=None,
    ignore_index=-100, reduction="mean",
    profile="balanced", backend="auto", config=None,
    gradient_materialization_scale=None, grad_scaler=None,
) -> DistillationLossOutput
```

### Positional inputs

| Argument | Type / shape | Contract |
|---|---|---|
| `student_hidden` | `(BT, H)` floating tensor | 2D; flattened `batch * seq` hidden states. Defines reference device. |
| `weight` | `(V, H)` floating tensor | Student output projection; `V >= 1`; same device and (outside autocast) same dtype as `student_hidden`. |
| `labels` | `(BT,)` `torch.long` | Values in `[0, V)` or exactly `ignore_index`; same device. Strict range validation by default (reports up to 5 offenders). |
| `teacher` | `TiedTeacher \| SeparateTeacher \| PrecomputedTeacher` | See [section 4](#4-teacher-descriptors). |

### Keyword parameters

| Keyword | Type | Default | Contract |
|---|---|---|---|
| `student_ce_weight` | `float` | `1.0` | Finite, `>= 0`. `bool` rejected. Weight of student CE term. |
| `teacher_ce_weight` | `float \| None` | `None` | Finite, `>= 0` when given. `None` resolves per mode: tied → `1.0`, separate/precomputed → `0.0`. `> 0` selects a trained path. |
| `kl_weight` | `float` | `0.0` | Finite, `>= 0`. `!= 0` activates the KL term. |
| `kl_temperature` | `float` | `1.0` | Finite, `> 0`; `1/T` and `T**2` must be normal float32 values. |
| `kl_teacher_temperature` | `float \| None` | `None` | `None` → equals `kl_temperature`. Same numeric constraints (`1/T_t` normal float32). Must equal `kl_temperature` for `mutual`/`cojeffreys`. |
| `kl_divergence` | `KLVariant` | `"forward"` | One of 15 variants ([variants.md §4](variants.md#4-kl-divergence-variants)). `mutual`/`cojeffreys` require tied or separate teacher (rejected for precomputed even when inactive). |
| `kl_alpha` | `float \| None` | `None` | Range depends on variant: `alpha_divergence` `(0,1)`; `renyi`/`renyi_reverse` `(0,2]` excluding `\|r−1\| < 0.01`; `dkd` `(0,16]`; `abkd` `[0.05,2]`. When the variant doesn't read it, a provided value must still be finite in `(0,1)`. When KL is inactive, the selected variant's own range is used for validation before canonicalizing to `0.5`. |
| `kl_beta` | `float \| None` | `None` | `skew_forward`/`skew_js`/`skew_reverse` `(0,1)`; `dkd` `(0,16]`; `abkd` `[0.05,2]`. Same canonicalization rule as `kl_alpha`; inactive KL still validates against the selected variant's `kl_beta` range. |
| `entropy_gate` | `EntropyGate \| None` | `None` | Read only by `entropy_adaptive` (defaults to `EntropyGate()` when unset); type-checked for every variant. |
| `ignore_index` | `int` | `-100` | Any integer (`bool` rejected). Never used as a memory index. |
| `reduction` | `"mean" \| "sum"` | `"mean"` | `mean` divides by the non-ignored count, clamped to `>= 1`. |
| `profile` | `"balanced" \| "fast" \| "debug"` | `"balanced"` | Preset `KernelConfig` base — see [section 6](#6-profiles). |
| `backend` | `"auto" \| "triton" \| "torch"` | `"auto"` | See [section 7](#7-backends). |
| `config` | `KernelConfig \| None` | `None` | Field-wise overrides merged on top of the profile — see [section 5](#5-kernelconfig). |
| `gradient_materialization_scale` | scalar tensor \| `None` | `None` | One-element `float32` tensor on the student device, without gradients. Multiplied into logit gradients inside the kernel as it materializes them and divided back out in backward, so gradients are unchanged. Guards an fp16 underflow that a loss-level `GradScaler` cannot reach — see [Gradient scaling under fp16](#gradient-scaling-under-fp16). Ignored by `backend="torch"`. Mutually exclusive with `grad_scaler`. |
| `grad_scaler` | `torch.amp.GradScaler \| None` | `None` | When enabled, derives `gradient_materialization_scale` from the scaler; a disabled scaler is a no-op. It does not scale the returned loss — call `scaler.scale(loss).backward()` as usual. See [Gradient scaling under fp16](#gradient-scaling-under-fp16). |

**Precedence:** for every field present on both, a non-`None` `KernelConfig` value overrides the call argument (`kl_weight`, `kl_temperature`, `kl_teacher_temperature`, `kl_divergence`, `kl_alpha`, `kl_beta`, `entropy_gate`), and a non-`None` `KernelConfig` value overrides the profile (`grad_weight_accumulation`, `max_fused_size`, `autotune`, `refresh_autotune`, `chunk_size`). `num_chunks` and `max_chunks` exist only on `KernelConfig`.

### Return value

`DistillationLossOutput`, a `NamedTuple`:

```python
class DistillationLossOutput(NamedTuple):
    loss: torch.Tensor        # weighted sum — backprop through this
    student_ce: torch.Tensor  # unweighted, detached
    teacher_ce: torch.Tensor  # unweighted, detached
    kl: torch.Tensor          # unweighted, detached
```

`loss = student_ce_weight·student_ce + teacher_ce_weight·teacher_ce + kl_weight·kl`. A component whose weight is `0` is never computed and reads exactly `0` (not its would-be value). Output dtype is `promote_types(input_dtype, float32)`.

**Autograd:** each fused Triton forward supports one backward pass (cached gradients consumed in place); recompute the loss instead of `retain_graph=True`. The Torch backend has no one-backward restriction, but `alpha_divergence`, `renyi`, `renyi_reverse`, and `abkd` use stabilized first-order gradient proxies. Requesting higher-order gradients for those four variants raises instead of silently returning a Hessian that omits KL curvature. All entry points are `torch._dynamo.disable`d.

### Gradient scaling under fp16

`torch.amp.GradScaler` alone cannot prevent underflow on the fused path. The reason is *when* each scale is applied.

An ordinary fp16 layer produces its gradients during `backward()`. `GradScaler` inflates the loss beforehand, so that inflation is already part of the chain rule by the time any gradient value is written — small values reach fp16 pre-multiplied and stay representable.

The fused kernel does not follow that order. It computes logit gradients **inside `forward()`**, in the same sweep that produces the loss, and stores them into a compute-dtype buffer (fp16 when inputs are fp16) to avoid materializing a second `(BT, vocab)` tensor. Its `backward()` only rescales that stored buffer. So:

1. `forward()` writes gradients to fp16; values below the subnormal floor flush to exactly zero.
2. The scaler's factor arrives later, in `backward()`.
3. Multiplying a stored zero by any factor is still zero — the signal is gone, not merely shrunk.

`gradient_materialization_scale` closes that gap by applying a factor at the only moment that helps: inside the kernel, on the value still held in a float32 register, *before* the fp16 store. Backward divides the same factor out, so the gradient reaching the parameter is numerically the fp32 result.

The two mechanisms are complementary rather than redundant. `GradScaler` protects every *other* layer in the model; `grad_scaler=` is the convenience path that reads the scaler's current factor and forwards it here as the materialization scale. Because both arguments fill the same slot, passing them together raises `ValueError`.

The effective factor also folds in token-mean reduction, and must stay finite and normal in the gradient storage dtype. When it would not, the kernel selects a numerically neutral factor on device and defers the reduction to backward — no host scan, no synchronization. Gradients stay correct, but the underflow guard is inactive for that call. `backend="torch"` differentiates through autograd without a materialized gradient buffer, so no scale applies there and gradients are bit-identical with or without one.

See [guide.md §4](guide.md#4-mixed-precision-and-gradient-scaling) for the training-loop recipes.

---

## 2. `DistillationLoss` (nn.Module)

Stores every keyword of `distillation_loss` at construction; `forward(student_hidden, weight, labels, teacher, gradient_materialization_scale=None, grad_scaler=None)` forwards the stored configuration plus the per-step gradient scale. No parameter is a learnable tensor; the module is a convenience container, and attributes may be reassigned between calls.

---

## 3. Hyperparameter objects

### `EntropyGate`

`EntropyGate(alpha_min=0.0, alpha_max=1.0, tau=0.5, scale=0.1)` — frozen dataclass for the `entropy_adaptive` gate `α(h) = alpha_min + (alpha_max − alpha_min) · sigmoid((h − tau) / scale)` where `h` is teacher entropy normalized by `log V` (`0` when `V == 1`).

Constraints: all fields finite floats (`bool` rejected), `0 <= alpha_min <= alpha_max <= 1`; both `scale` and `1 / scale` must be finite normal float32 values. The gate is computed from the detached teacher distribution and never receives gradient.

### `ChunkSize`

Immutable value object with three constructors:

- `ChunkSize.nrows(n)` — exactly `n` rows per chunk (`n` a positive int, clamped to `BT` at execution).
- `ChunkSize.fast()` — targets 1024 rows per chunk, clamped to at least 2 and at most 16 chunks.
- `ChunkSize.dynamic()` — initial partition chosen from the workload shape.

Raw integers, strings, `None`, and legacy non-positive sentinels are rejected wherever a `ChunkSize` is expected. `is_dynamic`, `is_fast`, and `value` are read-only accessors (`value` raises on a dynamic or fast instance).

---

## 4. Teacher descriptors

### `TiedTeacher(hidden, mode=None)`

- `hidden: (BT, H)` — must match the student shape exactly; the student's `weight` is shared.
- `mode: None | "trained" | "frozen" | "cotrained"` — a declarative consistency check, not a routing switch. The effective role derives from `teacher_ce_weight`, `kl_weight`, and `kl_divergence`; a contradiction raises `ValueError`:
  - `"frozen"` requires `teacher_ce_weight == 0` and no active `mutual`/`cojeffreys`;
  - `"trained"` requires `teacher_ce_weight > 0` and no active `mutual`/`cojeffreys`;
  - `"cotrained"` requires `kl_weight > 0` with `mutual`/`cojeffreys`.

### `SeparateTeacher(hidden, weight)`

- `hidden: (BT, D_t)`, `weight: (V, D_t)` — own projection; `D_t` may differ from `H`, `V` must match the student vocabulary.
- Gradients flow to whichever tensors have `requires_grad=True`, under the path's gradient policy (CE for trained, KL(+CE) for cotrained).

### `PrecomputedTeacher(logits=None, teacher_hidden=None, teacher_weight=None, trust_cache_immutability=False, logits_temperature=None)`

Validated, gradient-free teacher cache. Constraints enforced at construction:

- Exactly one representation: `logits: (BT, V)` or `teacher_hidden: (BT, D_t)` + `teacher_weight: (V, D_t)` (both together).
- No cached tensor may `require_grad` (use `SeparateTeacher` for a live teacher).
- `logits` values: finite or `-inf` (zero support); `NaN`/`+inf` rejected. `teacher_hidden`/`teacher_weight` values: strictly finite.
- Logit caches are row-centered in their source dtype before chunk-wise conversion; finite values below the compute dtype range are clamped while `-inf` support sentinels remain exact.
- A non-finite value produced by the cached hidden/weight GEMM is invalid and remains visible as `NaN` loss/active gradients rather than a dead cache row.
- Cached tensors are not cloned. A validation stamp (identity, layout, version counter) is captured at construction and re-verified on every call; normal in-place mutation after construction raises `ValueError`. Unsafe `.data` mutation is outside the guarantee.
- Construction performs full-content validation reductions; construct the descriptor once and retain it outside the training step.
- `trust_cache_immutability=True` is needed whenever `_version` is unavailable, including inference tensors created under `torch.inference_mode()` and tensor subclasses without a readable version counter.
- `logits_temperature` (default `None`) declares logits already tempered at that temperature (finite `> 0`, `logits` representation only), requires `teacher_ce_weight=0.0`, and makes the loss apply no further teacher temperature. When `kl_teacher_temperature` is left at its default the cache temperature is adopted; an explicitly passed `kl_teacher_temperature` that differs from it is rejected. `None` keeps raw logits, so the loss applies its own teacher temperature.

Shape checks against the student (`(BT, V)` / `(BT, D_t)`+`(V, D_t)`) happen at call time. The `logits` dtype may differ from the model dtype (converted chunk-wise); it must be floating.

`teacher.prepare_logits(dtype)` replaces that repeated chunk-wise conversion with an explicit persistent cache:

- `dtype` must be one of the Torch model dtypes (`float16`, `bfloat16`, `float32`, or `float64`). Use the dtype selected for the student model.
- An already contiguous logits cache with that dtype returns the same descriptor and allocates nothing. Otherwise the method row-centers before a narrowing cast, materializes a contiguous tensor, and returns a new `PrecomputedTeacher`.
- The new descriptor preserves `logits_temperature` and the normal validation stamp/mutation checks. The source descriptor remains independent.
- Preparation is never implicit: retain the returned descriptor for reuse and budget its full `(BT, V)` allocation. Hidden/weight caches are rejected because this method does not materialize a teacher projection.

---

## 5. `KernelConfig`

Frozen dataclass of per-call overrides. Every field defaults to `None`/"unset" and only overrides when set:

| Field | Type | Effect |
|---|---|---|
| `grad_weight_accumulation` | `"auto" \| "compute" \| "fp32" \| None` | Dtype of the chunked weight-gradient accumulator. `"auto"` (default) uses float32 when the parameter dtype differs from the compute dtype or the parameter is float16, and the compute dtype otherwise; `"compute"` always accumulates in the compute dtype; `"fp32"` always accumulates in float32. |
| `chunk_size` | `ChunkSize \| None` | `ChunkSize.nrows(n)` (fixed `n` rows), `ChunkSize.fast()` (1024 rows per chunk, at least 2 and at most 16 chunks), or `ChunkSize.dynamic()` (workload-shape heuristic). Unset inherits the profile value, falling back to `dynamic()`. Mutually exclusive with `num_chunks` only for `nrows`. |
| `num_chunks` | `int \| None` | Positive explicit request. Overrides `fast()`, `dynamic()`, and `max_chunks`. |
| `max_chunks` | `int \| None` | Lower-priority ceiling for automatic `fast()` and `dynamic()` partitioning; ignored by explicit `num_chunks` and `ChunkSize.nrows(n)`. Default is `16` for `fast()` and `min(32, 2·max(1, BT//512))` for `dynamic()`, which also bounds its OOM retry loop. |
| `max_fused_size` | `int \| None` | Power of two `<= 32768`; per-launch fused block ceiling. |
| `kl_weight`, `kl_temperature`, `kl_teacher_temperature`, `kl_divergence`, `kl_alpha`, `kl_beta`, `entropy_gate` | as in the call | Override the call arguments when set (same validation). |
| `autotune` | `bool \| "Max" \| None` | `False` = defaults only; `True` and `"Max"` use the same candidates allowed by `max_fused_size`, while `"Max"` measures finalists more deeply. Cached winners are honored by both modes. `"Max"` uses fixed screen, finalist, and robust-stage warmup/minimum/maximum controls; the corresponding environment overrides are ignored. |
| `refresh_autotune` | `bool \| None` | Re-benchmark once per workload and process run at the selected mode's measurement depth. Invalid when effective `autotune=False`. |
| `assume_valid_labels` | `bool \| None` | `True` skips the synchronous host label scan; kernel-side `ignore_index`-safe indexing stays active. |

Type errors are raised at construction (`chunk_size` must be a `ChunkSize` or `None`, flags must be `bool` or `None`, `grad_weight_accumulation` one of its literals or `None`, `entropy_gate` an `EntropyGate` or `None`).

## 6. Profiles

| Profile | `max_fused_size` | `autotune` | `grad_weight_accumulation` | `chunk_size` |
|---|---|---|---|---|
| `balanced` | `16384` | `True` | `"auto"` | unset → `dynamic()` |
| `fast` | `32768` | `"Max"` | `"auto"` | `fast()` |
| `debug` | `32768` | `False` | `"fp32"` | unset → `dynamic()` |

A `KernelConfig` field always wins over its profile value. `num_chunks` and `max_chunks` exist only on `KernelConfig`.

`ChunkSize.fast()` targets throughput: the partition ignores vocabulary size, stays at or below 16 chunks (chunks widen past `16·1024` rows rather than multiplying), and is never retried. An out-of-memory failure raises with the partition it used and a suggested `num_chunks`, which the caller sets to choose a new partition.

## 7. Backends

| `backend` | Behavior |
|---|---|
| `"auto"` | Triton when Triton is importable, tensors are CUDA/HIP, the dtype is supported, and the policy has a Triton specialization; otherwise the Torch reference plus a one-time process-wide `RuntimeWarning` naming the reason. |
| `"triton"` | Forced fused path. Raises `RuntimeError` (no Triton/CUDA), `ValueError` (unsupported dtype; precomputed teacher CE), or `UnsupportedTritonPolicyError` (valid policy without a Triton specialization). |
| `"torch"` | Reference implementation — numerical source of truth, CPU or GPU, materializes full `(BT, V)` logits. Autograd is standard except the four stabilized proxy variants documented above, which are first-order only. |

Dtype support: Torch `{fp16, bf16, fp32, fp64}`; Triton `{fp16, bf16, fp32}` with bf16 requiring native hardware support. Mixed dtypes are only accepted under autocast (resolved to the autocast dtype).

`is_available()` returns `True` when Triton + CUDA/HIP kernels can be selected at all (does not check dtype or policy).

---

## 8. Low-level ops

`orda_kernel.ops.tied_teacher.tied_distillation_loss`, `ops.separate_teacher.separate_distillation_loss`, and `ops.precomputed_teacher.precomputed_distillation_loss` expose the mode slices directly. Differences from `distillation_loss`:

- Return a plain 4-tuple `(loss, student_ce, teacher_ce, kl)`.
- Triton-only execution model (no `backend`/`profile` arguments); they use the same planner and execution engine, so they get the same validation, OOM retry, and chunk caching.
- `autotune` defaults to `False` (the high-level profiles default it on).
- Low-level APIs expose `assume_valid_labels: bool = False` for the trusted-label fast path.
- `teacher_ce_weight=None` defaults: tied `1.0`, separate `0.0`, precomputed `0.0`.
- `precomputed_distillation_loss` additionally accepts either raw cache tensors (`logits_teacher` or `teacher_hidden`+`teacher_weight`) or a validated descriptor via keyword `teacher=PrecomputedTeacher(...)`, plus `assume_valid_teacher_cache: bool = False` to skip per-call stamp re-verification.

Autograd `Function` classes and kernel helpers below these functions are private implementation API.

---

## 9. Aligned-span API

### `aligned_span_distillation_loss`

```python
aligned_span_distillation_loss(
    student_hidden, student_weight, student_labels,
    teacher,            # CrossVocabTeacher | PrecomputedSpanTeacher
    span_pairs,         # AlignedSpanPairs
    config=None,        # AlignedLikelihoodConfig
    *, profile="balanced", backend="auto", kernel_config=None,
    gradient_materialization_scale=None, grad_scaler=None,
) -> DistillationLossOutput
```

- `student_hidden: (BT_s, H)`, `student_weight: (V_s, H)` (`V_s >= 1`), `student_labels: (BT_s,) torch.long`, all on one device.
- `backend="triton"` requires a `PrecomputedSpanTeacher` plus Triton and a supported CUDA/HIP dtype; a `CrossVocabTeacher` always runs on the Torch reference (`auto` falls back silently — it is the reference teacher type).
- `kernel_config` accepts `grad_weight_accumulation` plus chunking/autotune/label fields; the tokenwise KL fields (`kl_weight`, `kl_temperature`, `kl_teacher_temperature`, `kl_divergence`, `kl_alpha`, `kl_beta`, `entropy_gate`) are rejected.
- `gradient_materialization_scale` and `grad_scaler` follow the tokenwise contract above and are mutually exclusive.
- Returns `teacher_ce = 0` always; `kl` holds the reduced binary span divergence.

### `AlignedSpanPairs(student_ranges, teacher_ranges)`

Two `(C, 2)` int32/int64 tensors of half-open `[start, end)` ranges, equal `C`, validated against batch bounds at call time (`0 <= start < end <= rows`).

### `AlignedLikelihoodConfig`

| Field | Default | Contract |
|---|---|---|
| `student_ce_weight` | `1.0` | finite, `>= 0` |
| `span_kl_weight` | `1.0` | finite, `>= 0` |
| `ignore_index` | `-100` | int |
| `reduction` | `"mean"` | `"mean"` or `"sum"` (span mean counts only kept spans) |
| `objective` | `"binary_kl"` | or `"binary_ce"` |

### `CrossVocabTeacher(hidden, weight, labels)`

Live teacher over its own vocabulary (`hidden: (BT_t, D_t)`, `weight: (V_t, D_t)`, `labels: (BT_t,) torch.long`), same device as the student tensors. Span likelihoods are computed under `no_grad`; the teacher is always detached. Ignored rows are masked before projection; a non-finite GEMM result on an active row is rejected.

### `PrecomputedSpanTeacher(span_log_probs, trust_cache_immutability=False, teacher_ranges=None, trust_teacher_ranges=False)`

1D float tensor of per-span log-likelihoods: values `<= 0` or `-inf` (zero-support), no `NaN`/`+inf`, `requires_grad` forbidden, entry count must equal `C`. Same validation-stamp mutation guard as `PrecomputedTeacher`.

`teacher_ranges` binds the cache to the `(C, 2)` ranges it was built from (`build_precomputed_span_teacher` fills it in); at call time it is compared against `AlignedSpanPairs.teacher_ranges` and a mismatch raises. A raw self-built cache must either supply `teacher_ranges` or opt out with `trust_teacher_ranges=True` (mutually exclusive with `teacher_ranges`).

### `build_precomputed_span_teacher(teacher_hidden, teacher_weight, teacher_labels, teacher_ranges, *, ignore_index=-100)`

Materializes a detached `PrecomputedSpanTeacher` in bounded chunks (~`2**24` logit elements per chunk, fp32 statistics or fp64 for fp64 inputs). Validates shapes, dtypes, device uniformity, label range, range bounds, and rejects any span with zero valid teacher tokens, reporting the offending span indices.

---

## 10. Precomputed-input builders

All four helpers are Torch-side preparation for `PrecomputedTeacher`: they detach their outputs and never select a kernel path or affect autotune identity.

### `build_exact_token_map(student_vocab, teacher_vocab) -> Tensor[V_t] int64`

Exact-string match from each teacher token to a student index, `-1` when unmatched. Duplicate student strings resolve to the first occurrence. All entries must be `str`.

### `project_teacher_logits(teacher_logits, token_map, student_vocab_size, *, unmatched="renormalize", temperature=1.0) -> Tensor[N, V_s]`

Projects `[N, V_t]` teacher logits onto student columns; duplicate teacher tokens sum their probability mass; unmatched student columns receive `-inf`, so their softmax probability is exactly zero at every temperature and survives a downstream fp16 cast.

- `unmatched="renormalize"`: matched mass rescaled to one (proper log-probabilities).
- `unmatched="mask"`: keeps the absolute matched log-mass (grouped log-sum-exp).
- `temperature=T` (default `1.0`, finite `> 0`) divides the raw teacher logits before mass is aggregated. Temperature does not commute with many-to-one aggregation, so the output is bound to `T`: declare it with `PrecomputedTeacher(logits=..., logits_temperature=T)` so the loss adopts `T` as the teacher temperature and does not temper a second time.

Validation: `token_map` 1D int32/int64 with one entry per teacher column, values `>= -1`, at least one match, mapped indices `< student_vocab_size`; `teacher_logits` a floating tensor free of `NaN`/`+inf` (`-inf` allowed), and every row needs at least one finite logit on a matched column. Half dtypes compute in fp32. Projection stays in log space, and finite source logits remain finite after narrowing even when their probability is negligible. Output is detached.

### `mix_teacher_logits(logits_list, weights, *, mode="arithmetic") -> Tensor[N, V]`

Mixes several detached teachers over one shared vocabulary.

- Inputs: `>= 1` teacher, all `[N, V]` with identical shape/device/dtype, no `NaN`/`+inf`, no all-dead rows. `weights`: 1D, one per teacher, finite, strictly `> 0` (normalized internally in log space).
- `mode="arithmetic"`: weighted average of probabilities → normalized log-probabilities.
- `mode="geometric"`: product-of-experts weighted log-score; zero support in any teacher stays zero, and every row must keep at least one commonly supported token (else `ValueError`).
- Geometric mixing and `generalized_jensen_shannon` require every normalized weight to remain representable in the compute dtype; arithmetic mixing keeps extreme weights in log space.
- Compute dtype fp32 (fp64 in → fp64); output detached.

### `generalized_jensen_shannon(logits_list, weights) -> Tensor[N]`

Detached per-row teacher-agreement diagnostic (weighted mean of each teacher's KL to the arithmetic mixture), clamped `>= 0`. Same input validation as `mix_teacher_logits`.

---

## 11. Runtime controls

- `is_available() -> bool` — Triton importable and `torch.cuda.is_available()`.
- `reset_autotune_cache()` — clears in-memory and on-disk autotune state.
- `begin_autotune_run()` — starts a fresh process-local run (per-run tuning state) without evicting persisted winners.
- `__version__` — distribution version, `"0+unknown"` outside an installed package.

## 12. Autotune environment variables

Winners are cached in an in-memory LRU and a SQLite (WAL) store, keyed by device fingerprint, workload shape/pipeline, dtype, and a timed-source AST digest. Cache root: `ORDA_AUTOTUNE_CACHE_DIR`, else `%LOCALAPPDATA%/orda_kernel` (Windows), `$XDG_CACHE_HOME/orda_kernel`, or `~/.cache/orda_kernel`. On clusters point it at node-local storage — SQLite/OS locking is not reliable on every NFS; an unusable store warns once per path and degrades to process-local autotuning.

Out-of-range values are clamped with a `RuntimeWarning`; non-integers fall back to default.

| Variable | Default | Bounds | Meaning |
|---|---|---|---|
| `ORDA_AUTOTUNE_CACHE_DIR` | platform cache dir | — | Store location override. |
| `ORDA_AUTOTUNE_MEMORY_ENTRIES` | `4096` | `[1, 65536]` | In-memory winner LRU size. |
| `ORDA_AUTOTUNE_DISK_ENTRIES` | `4096` | `[1, 65536]` | Disk store LRU size. |
| `ORDA_AUTOTUNE_LEASE_SECONDS` | `60` | `[10, 3600]` | Cross-process device-lease lifetime. |
| `ORDA_AUTOTUNE_LEASE_POLL_MS` | `100` | `[25, 5000]` | Lease wait poll interval. |
| `ORDA_AUTOTUNE_LEASE_WAIT_SECONDS` | `2` | `[1, 3600]` | Max wait for a busy device lease; on timeout the call uses the latest cached or default config instead of benchmarking. |
| `ORDA_AUTOTUNE_SQLITE_TIMEOUT_SECONDS` | `15` | `[1, 120]` | SQLite busy timeout. |
| `ORDA_AUTOTUNE_SQLITE_INIT_RETRIES` | `3` | `[1, 10]` | Schema init attempts. |
| `ORDA_AUTOTUNE_SQLITE_INIT_RETRY_MS` | `50` | `[1, 1000]` | Delay between init attempts. |
| `ORDA_AUTOTUNE_SCREEN_WARMUP_REPS` | `2` | `[0, 2]` | Screening warmup reps per candidate. |
| `ORDA_AUTOTUNE_SCREEN_REPS` | `3` | `[3, 5]` | Screening timed reps. |
| `ORDA_AUTOTUNE_FINALISTS` | `3` | `[1, 3]` | Candidates advanced to robust timing. |
| `ORDA_AUTOTUNE_ROBUST_MIN_REPS` | `4` | `[1, 8]` | Robust-stage minimum reps. |
| `ORDA_AUTOTUNE_ROBUST_MAX_REPS` | `12` | `[1, 12]` | Robust-stage maximum reps. |
| `ORDA_AUTOTUNE_ROBUST_TARGET_MS` | `80` | `[1, 500]` | Robust-stage timing budget. |
| `ORDA_AUTOTUNE_TIMING_BATCH` | `1` | `[1, 16]` | Launches per timing sample. |
| `ORDA_AUTOTUNE_FAILURE_BACKOFF_MS` | `1000` | `[10, 60000]` | Base backoff after a transient benchmark failure. |
| `ORDA_AUTOTUNE_FAILURE_BACKOFF_MAX_MS` | `30000` | `[10, 600000]` | Backoff ceiling (never below the base). |

## 13. Exception summary

| Exception | Raised for |
|---|---|
| `ValueError` | Almost every contract violation: shapes, dtypes, devices, weight/temperature/hyperparameter ranges, label range, descriptor invariants, cache mutation, `mutual`/`cojeffreys` with `PrecomputedTeacher`, precomputed teacher CE on `backend="triton"`, chunk-request conflicts. |
| `TypeError` | Wrong object kinds: non-tensor descriptor fields, non-`ChunkSize` chunk requests, non-`EntropyGate` gates, non-`AlignedSpanPairs`/`AlignedLikelihoodConfig` arguments. |
| `UnsupportedTritonPolicyError` (subclass of `ValueError`) | `backend="triton"` with a semantically valid policy that no Triton specialization executes. Not exported at the package root — catch `ValueError`; the common cases (precomputed teacher CE, live `CrossVocabTeacher`) are pre-empted by plain `ValueError` messages. |
| `RuntimeError` | `backend="triton"` without Triton/CUDA-HIP; OOM once the automatic partition cannot split further (message includes shape, the partition used, and remediation as `num_chunks`); internal plan inconsistencies. |
| `RuntimeWarning` | One-time `auto`→Torch fallback notice; autotune store/environment degradations. |
