# Designer guide

This document is for people changing the kernel: adding a KL variant, a teacher mode, a launch pipeline, or touching the canonical Triton kernel. It records design tenets, contracts that must not drift, and checklists for extending them. For runtime behavior see [architecture.md](architecture.md); for user-facing contract see [reference.md](reference.md) and [variants.md](variants.md).

---

## 1. Design tenets

1. **One numerical body, many vertical slices.** One launched Triton kernel (`ops/_kernels/ce_kl_fwdbwd.py::ce_kl_fwdbwd_kernel`); `_kl_math.py` holds inline `@triton.jit` helpers called only inside it, never launched (`test_source_contracts.py` pins the single-launch rule). Each teacher mode keeps a vertical slice — benchmark task, chunk loop, GEMMs, autograd `Function`, and fixed gradient ABI live together in one mode file. Mode-specific `Function`s and benchmark builders are intentional duplication: they protect fixed gradient ownership, readable slices, and specialized launch pipelines.
2. **Centralize what must not fork.** Numerical CE/KL code, launch-request construction, validation, and execution planning are single-sourced. A bug fixed in `_kl_math.py` or `_torch_backend.py` is fixed for every mode.
3. **Policy is cold-path Python; plans are immutable.** All routing decisions happen once per call in `_planning/`; execution receives frozen `ExecutionPlan` / `ChunkAttempt` objects and never re-resolves policy.
4. **The capability registry is the product boundary.** The canonical kernel can compile more constexpr combinations than the product supports. `_planning/capabilities.py` decides what is reachable; everything else is rejected before launch.
5. **The Torch reference is the source of truth.** Every kernel behavior — including edge-case pinning (dead teacher rows, zero-support sentinels) — must be mirrored in `_torch_backend.py`, because tests validate Triton against it and CPU-only users execute it directly.

## 2. Layer map

```text
api.py / ops/{tied,separate,precomputed}_teacher.py / ops/precomputed_inputs/aligned_span.py
        │   validation (_validation.py, _scalar_validation.py), config merge (_config.py)
        ▼
_planning/            policy.py: weights+variant → kernel path + specialization (immutable plan)
        │             capabilities.py: product boundary; models.py: vocabulary + hyperparam specs
        ▼
_execution/           dispatcher.py: chunk loop, OOM retry, partition cache
        │             aligned_span.py: two-stage span executor
        ▼
ops/_kernels/<mode>/  vertical slice: GEMMs, launches, autograd Function, gradient ABI
        ▼
ops/_kernels/ce_kl_fwdbwd.py   the only launched Triton kernel (+ inline _kl_math.py @triton.jit helpers)

_autotune/            full-batch benchmark, winner selection, SQLite/WAL store, leases
```

## 3. Canonical kernel contract

### 3.1 Constexpr axes

Semantic specialization axes of `ce_kl_fwdbwd_kernel`:

| Axis | Meaning |
|---|---|
| `TIED_LAYOUT` | Student and teacher rows stacked in one buffer vs independent buffers. |
| `COMPUTE_TEACHER_CE` | Teacher CE term active. |
| `COMPUTE_KL` | KL term active. |
| `TEACHER_KL_GRAD` | Teacher receives KL gradient (`mutual`/`cojeffreys` only). |
| `BOTH_T_ONE` | Both resolved temperatures are `1.0` (skip scaling). |
| `KL_STYLE` | One of the 15 KL codes below. |
| `RENYI_NEAR_ONE` | Rényi order within `0.125` of one, enabling stable control-variate sweep. |
| `KERNEL_STAGE` | `TOKENWISE=0`, `ALIGNED_STATS=1`, `ALIGNED_GRAD=2`. |
| `BLOCK_SIZE` | Launch tile width (autotuned). |

KL style codes (`kl_style_of`): `forward=0`, `jeffreys=1`, `mutual=2`, `cojeffreys=3`, `reverse=4`, `alpha_divergence=5`, `skew_forward=6`, `skew_js=7`, `entropy_adaptive=8`, `skew_reverse=9`, `renyi=10`, `renyi_reverse=11`, `dkd=12`, `abkd=13`, `triangular=14`.

The reachable product policy (15 styles across the mode matrix) produces 170 legacy kernel identities. The canonical kernel, including its two aligned stages, reduces these to 165 semantic combinations — `test_kernel_specializations.py` measures and pins both counts. If a change moves either number, that is a product decision, not a refactor.

A `TEACHER_SOURCE` axis is intentionally absent: precomputed and GEMM-produced teacher logits have identical kernel-side memory semantics; splitting them would duplicate binaries without changing generated code.

### 3.2 Runtime scalars vs specialization identity

`kl_alpha`, `kl_beta`, `entropy_gate`, and both temperatures remain plan scalars rather than direct specialization values. Unused values canonicalize (`0.5`, gate `None`, temperatures `1.0` when KL is inactive) so equivalent plans compare equal. Only bounded predicates derived from those scalars enter the kernel identity: `BOTH_T_ONE` for temperature scaling and `RENYI_NEAR_ONE` for the numerically stable Rényi path. Putting a continuous scalar into the specialization would explode compile counts and autotune identities.

### 3.3 Pointer layout

- `TIED_LAYOUT=True`: `S_ptr` holds student rows followed by teacher rows; the kernel addresses teacher row `i` as `i + n_rows`; `T_ptr` is ignored.
- `TIED_LAYOUT=False`: `S_ptr` and `T_ptr` are independent row-major logit buffers (shared by separate and precomputed modes).
- When a constexpr path cannot read or write a teacher pointer, the wrapper may alias it to an active buffer; the disabled branch is removed at compile time. Never pass a null/undersized pointer.
- `KERNEL_STAGE=ALIGNED_STATS` reuses `L_ptr` for NLL output and `T_L_ptr` for saved student LSE. `ALIGNED_GRAD` reads LSE from `T_L_ptr` and per-row coefficient from `K_ptr`. No stage-specific pointer may expand the canonical signature.

### 3.4 Label safety invariant

Strict mode checks label ranges on the host; `assume_valid_labels=True` only removes that CUDA-to-host synchronization. The kernel must always compute

```python
y_safe = tl.where(ignored | (~in_range), 0, y)
```

and use `y_safe` for every label-based load and comparison. `ignore_index` (e.g. `-100`) is valid API input but must never reach memory indexing. Do not add a code path that indexes with a raw label, even behind the trusted flag.

### 3.5 Numerical edge contracts

These behaviors are part of the public contract and are pinned by differential tests; the kernel and the Torch reference must move together:

- Zero-preserving multiplication for `p·(log p − log q)` terms: `p == 0` contributes exactly `0` in value and gradient.
- Support-safe reverse-direction rows: `p > 0` where `q == 0` yields a `+inf` row with a zero row gradient.
- Dead precomputed teacher rows (all `-inf`): teacher CE and KL pinned to `0`; the row still counts in the `mean` denominator.
- `-inf` is a support sentinel only for raw cached logits. Non-finite logits produced by tied/separate/cached-hidden GEMMs surface as `NaN` loss and active gradients instead of being reclassified as dead rows.
- Rényi/ABKD support instability yields `+inf` with a zero KL row gradient; invalid teacher numerics follow the source-aware `NaN` contract above.
- The fused kernel writes exact zero to ignored logits-gradient rows. Internal weight-gradient GEMMs trust that contract and mask only hidden rows, which is still required because IEEE arithmetic makes `0 * NaN` / `0 * Inf` non-finite. The generic helper keeps masking both operands unless that trusted path is selected explicitly.
- DKD: TCKD live at either teacher-target endpoint; NCKD zero only without teacher non-target support; missing student support → `+inf` KL row, zero KL gradient, CE unchanged.

## 4. Autograd ownership ABI

Eight `torch.autograd.Function` classes keep fixed signatures and gradient slots:

| Function | Inputs after `ctx` | Teacher gradient ownership |
|---|---:|---|
| `TiedFrozenCEFunction` | 7 | none |
| `TiedTrainedCEFunction` | 7 | teacher CE through shared weight/hidden |
| `TiedCotrainedCEFunction` | 7 | teacher KL through shared weight/hidden, optionally teacher CE |
| `PrecomputedCEFunction` | 9 | none |
| `PrecomputedAlignedSpanFunction` | 8 | none; student hidden and weight only |
| `SeparateFrozenCEFunction` | 8 | none |
| `SeparateTrainedCEFunction` | 8 | teacher CE |
| `SeparateCotrainedCEFunction` | 8 | teacher KL, optionally teacher CE |

Rules:

- Inactive objective branches return `None` for inputs outside the graph.
- Cached gradient buffers are scaled in place during backward; dtype conversion allocates only when the output dtype differs. This consumes the buffers — one backward per forward is a contract, not a bug.
- Do not merge Functions across modes, and do not let a helper below a Function reach into another mode's slice. Share helpers below the Function boundary only.

## 5. Aligned-span pipeline design

The span objective cannot finish in one vocabulary sweep: a span coefficient depends on every token NLL in the span, and each token's logits gradient needs that coefficient. Hence two kernel stages around an FP64 host-side reduction:

1. Chunked student GEMM + `ALIGNED_STATS` emits FP32 token NLL and row LSE.
2. FP64 prefix/difference-array math reduces spans and produces one scalar coefficient per token row.
3. `ALIGNED_GRAD` converts saved LSE + coefficient into `a_i * (p − onehot)` written back over the logits buffer.

Design constraints to preserve:

- One-chunk runs reuse the stage-1 logits buffer; multi-chunk runs recompute chunk-local logits so peak vocabulary memory stays chunk-local.
- Spans with no valid student token are excluded from loss and mean denominator; the precomputed builder refuses evidence-free teacher spans.
- The FP64 workspace is `O(BT + spans)` and is not reduced by chunking; its OOM must stay distinguishable from retryable chunk-vocabulary OOM.
- `binary_kl` and `binary_ce` share the fused stages and the logits gradient, therefore they must keep sharing one autotune winner.

## 6. Chunking and OOM recovery

- `ChunkPolicy` (initial partition, retry ceiling) is resolved at planning; the dispatcher owns the retry loop: on a retryable CUDA/HIP OOM it frees allocator blocks, doubles the chunk count, retries up to `max_chunks_limit`, and caches the successful partition in a bounded LRU keyed by tensor signatures plus specialization identity.
- The low-level ops route through the same engine — never give a direct API a private execution loop.
- Only allocation failures matching the OOM classifier are retryable; everything else propagates.

## 7. Autotune identity

Autotune benchmarks full-batch pipelines, not isolated kernel calls. A frozen `FullBatchLaunchRequest` carries all launch and workload identity into one selection call per autograd Function.

The persistent store (SQLite, WAL, schema version 1) keys winners by:

- device/toolchain fingerprint;
- workload and pipeline keys;
- an AST digest of every timed source module, normalized to be stable across the supported CPython versions (3.10–3.13: docstrings stripped, version-variant AST fields such as `type_params` removed);
- generation fencing and expiring cross-process device leases.

Consequences for changes:

- Touching the canonical kernel, a mode pipeline, runtime math helpers, the dispatcher, or the Torch benchmark reference changes the timed-contract digest and forces a new autotune identity. That is intended — stale winners must not survive a numerical change.
- Source-less wheels use a checked-in digest guarded by CI: regenerate the fallback digest whenever the timed surface changes.
- One process benchmarks a physical device at a time (bounded lease wait); contention falls back to the latest winner or default config rather than blocking training. Schema recovery revalidates the database under a per-store lock before replacing owned files.

## 8. Extension checklists

### 8.1 Any extension

1. Add product capability first in `_planning/capabilities.py`; unsupported combinations must be rejected at planning, not at launch.
2. Update the canonical kernel and the Torch reference together.
3. Keep per-mode autograd signatures fixed; share helpers below the Function.
4. Add CPU oracle tests, CUDA differential tests, ABI tests, specialization-count tests, and autotune contract coverage.
5. Regenerate the fallback timed-contract digest if the timed surface changed.

### 8.2 Adding a KL variant

1. Only add a style whose loss and both gradient directions are defined (student always; teacher direction at least specified as detached).
2. Extend `VALID_KL_VARIANTS`, the `KLVariant` literal, and — if it takes hyperparameters — `KL_HYPERPARAM_SPECS` with exact open/closed bounds and canonical defaults.
3. Assign the next `KL_STYLE` code; implement the Triton branch in the canonical kernel and the row helper in `_torch_backend.py`, including the zero-support and sentinel policy (section 3.5).
4. Decide detached vs teacher-KL-gradient. A teacher-gradient variant joins `TEACHER_KL_GRAD_VARIANTS`, is restricted to tied/separate cotrained paths, and must reject asymmetric temperatures.
5. Update the pinned specialization counts and the docs tables ([variants.md §4](variants.md#4-kl-divergence-variants), [README §KL variants](../README.md#kl-variants)).

### 8.3 Adding a teacher mode / pipeline

1. New vocabulary in `_planning/models.py`, capability entry, routing in `resolve_kernel_path`, gradient policy in `teacher_grad_policy`.
2. A new vertical slice under `ops/_kernels/<mode>/` with its own autograd `Function` (fixed ABI) and benchmark builder; reuse the canonical kernel — do not add a second `@triton.jit` body.
3. A public low-level op module exporting only the `*_distillation_loss` function.
4. Wire the dispatcher executor resolution and the chunk-cache key if the tensor set differs.

## 9. Test gates

| Gate | Pins |
|---|---|
| `test_kernel_specializations.py` | 170 legacy identity / 165 canonical semantic combination counts. |
| CPU oracle suites | Torch reference vs fp64 references, all variants and edge sentinels. |
| `test_public_api_cuda.py`, `test_low_level_ops_cuda.py` | Triton vs Torch differential: every teacher path, KL family, inactive branches, ignored non-finite rows (strict + trusted), noncontiguous inputs, precomputed dtype conversion, aligned-span stages. |
| `test_dlogits_contract_cuda.py` | Exact-zero ignored-row logits-gradient contract consumed by the hidden-only weight-gradient masking path, across every teacher mode and aligned-span execution. |
| `test_cartesian_cuda.py` (sharded) | Full public matrix: weights, label failures, layouts, chunk policies, APIs, dtypes. |
| `test_autotune_full_batch_cuda.py` | Bounded full-plan timing contract, 16K/32K launches, `autotune="Max"` coverage. |
| Multiprocess cache gate (manual) | Cold WAL/schema locking with 8 spawn workers, lease ownership, deterministic OOM-retry double. |

See [gpu-validation.md](gpu-validation.md) for the command matrix and the repository workflow files for the automated CI scope.
