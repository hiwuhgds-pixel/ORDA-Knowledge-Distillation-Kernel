# Execution variants

This document enumerates the variants the package executes: how a call routes to one of eight kernel paths, what each path supports, the contract of all fifteen KL divergence variants, and cross-cutting limits. Companion documents: [reference.md](reference.md) (per-parameter API contract) and [architecture.md](architecture.md) (internal organization).

Notation: `p` is the student distribution, `q` the teacher distribution, `BT` the flattened `batch * seq` row count, `H` the student hidden size, `D_t` the teacher hidden size, `V` the vocabulary size. `T_s` is `kl_temperature`, `T_t` is `kl_teacher_temperature`.

---

## 1. How a call resolves to a variant

Routing is deterministic and derives from three inputs: the teacher descriptor type, the loss weights, and the KL variant.

### 1.1 Teacher descriptor → teacher mode

| Descriptor | Teacher mode |
|---|---|
| `TiedTeacher(hidden, mode=None)` | `tied` |
| `SeparateTeacher(hidden, weight)` | `separate` |
| `PrecomputedTeacher(logits=… \| teacher_hidden=…, teacher_weight=…)` | `precomputed` |
| `CrossVocabTeacher` / `PrecomputedSpanTeacher` (aligned-span API only) | `precomputed` (aligned-span path) |

### 1.2 Teacher mode + weights + variant → kernel path

Evaluated in this order:

1. `kl_divergence ∈ {"mutual", "cojeffreys"}` and `kl_weight != 0` → the cotrained path for that mode.
2. Otherwise `teacher_ce_weight > 0` → the trained path.
3. Otherwise → the frozen path.
4. `precomputed` mode always resolves to `precomputed_tokenwise`.

`teacher_ce_weight=None` (the default) resolves per mode before routing:

| Mode | Default `teacher_ce_weight` |
|---|---|
| `tied` | `1.0` |
| `separate` | `0.0` |
| `precomputed` | `0.0` |

The same defaults apply to the low-level ops (`tied_distillation_loss`, `separate_distillation_loss`, `precomputed_distillation_loss`).

### 1.3 Kernel path → teacher gradient policy

| Kernel path | Teacher gradient policy |
|---|---|
| `tied_frozen`, `separate_frozen`, `precomputed_tokenwise` | `none` |
| `tied_trained`, `separate_trained` | `ce` (teacher CE only) |
| `tied_cotrained`, `separate_cotrained` | `kl`, or `ce_kl` when `teacher_ce_weight != 0` |
| `precomputed_aligned_span` | `none` (student hidden and weight only) |

An explicitly declared `TiedTeacher.mode` never changes routing; it asserts that the resolved role matches the declaration (section 2.1).

---

## 2. Tokenwise kernel paths

All seven tokenwise paths share the shape contract `student_hidden: (BT, H)`, `weight: (V, H)`, `labels: (BT,) torch.long`, all tensors on one device, plus the shared limits in section 5.

### 2.1 `tied_frozen`

- **Selected when:** `TiedTeacher`, `teacher_ce_weight == 0`, no active `mutual`/`cojeffreys` KL.
- **Teacher tensors:** `hidden: (BT, H)` — must match the student shape exactly; the projection `weight` is shared.
- **Gradients:** student hidden and shared weight only. The teacher hidden may `require_grad`, but this path contributes no gradient to it.
- **Active KL variants:** all thirteen teacher-detached variants. `mutual`/`cojeffreys` with `kl_weight != 0` re-route to `tied_cotrained`.
- **Inactive KL (`kl_weight == 0`):** any of the fifteen variants is accepted; the plan canonicalizes the variant to `forward` and both temperatures to 1.
- **Declared mode check:** `TiedTeacher(mode="frozen")` raises `ValueError` when `teacher_ce_weight != 0` or an active `mutual`/`cojeffreys` KL is configured.

### 2.2 `tied_trained`

- **Selected when:** `TiedTeacher` and `teacher_ce_weight > 0` (explicitly, or through the tied default of `1.0`), no active `mutual`/`cojeffreys`.
- **Gradients:** teacher CE flows to the teacher hidden and the shared weight, in addition to all student gradients. The KL term stays teacher-detached.
- **Declared mode check:** `TiedTeacher(mode="trained")` requires `teacher_ce_weight > 0` and no active teacher-KL-gradient variant.

### 2.3 `tied_cotrained`

- **Selected when:** `TiedTeacher`, `kl_divergence ∈ {"mutual", "cojeffreys"}`, `kl_weight != 0`.
- **Active KL variants:** only `mutual` and `cojeffreys`. There is no inactive form — with `kl_weight == 0` the call routes to frozen/trained.
- **Gradients:** the teacher hidden receives KL gradient (plus CE gradient when `teacher_ce_weight != 0`), and the shared projection weight accumulates gradient from both the student and teacher halves of the stacked logits buffer.
- **Limits:** `kl_teacher_temperature` must equal `kl_temperature`; asymmetric temperatures are defined only for teacher-detached variants.
- **Declared mode check:** `TiedTeacher(mode="cotrained")` requires `kl_weight > 0` with `kl_divergence ∈ ("mutual", "cojeffreys")`.

### 2.4 `separate_frozen`

- **Selected when:** `SeparateTeacher`, `teacher_ce_weight == 0` (the separate default), no active `mutual`/`cojeffreys`.
- **Teacher tensors:** `hidden: (BT, D_t)`, `weight: (V, D_t)`. `D_t` may differ from `H`; `V` must equal the student vocabulary.
- **Gradients:** none reach the teacher; teacher logits materialized chunk-wise and consumed detached.
- **Active/inactive KL variants:** same sets as `tied_frozen`.

### 2.5 `separate_trained`

- **Selected when:** `SeparateTeacher` and `teacher_ce_weight > 0`.
- **Gradients:** teacher CE owns teacher gradients — flow to whichever of `teacher.hidden` / `teacher.weight` has `requires_grad=True`. Every teacher-detached KL variant sends no KL gradient to teacher.

### 2.6 `separate_cotrained`

- **Selected when:** `SeparateTeacher`, `kl_divergence ∈ {"mutual", "cojeffreys"}`, `kl_weight != 0`.
- **Gradients:** `mutual`/`cojeffreys` KL owns teacher gradients, with optional teacher CE contribution (`ce_kl` policy).
- **Limits:** equal temperatures required, as with `tied_cotrained`.

### 2.7 `precomputed_tokenwise`

- **Selected when:** the teacher is a `PrecomputedTeacher`, always.
- **Representations (exactly one):**
  - `logits: (BT, V)` — cached teacher logits. Values must be finite or `-inf` (zero support; `NaN`/`+inf` rejected at construction). Dtype may differ from model dtype; cached logits row-centered in source dtype, then converted chunk-wise. Finite underflow clamped to compute range while `-inf` stays exact.
  - `teacher_hidden: (BT, D_t)` + `teacher_weight: (V, D_t)` — recompute teacher logits from cached trunk output. Values must be strictly finite; non-finite GEMM result surfaced as invalid numeric row.
- **Gradient contract:** cached tensors must not `require_grad` (`ValueError` at construction). No gradient ever reaches precomputed teacher; no trained/cotrained precomputed path exists.
- **Cache-mutation guard:** descriptor snapshots validation stamp (object id, data pointer, shape, stride, storage offset, dtype, device, layout, `requires_grad`, version counter) at construction and re-checks on every call. In-place mutation after construction raises `ValueError`; construct new descriptor instead. `trust_cache_immutability=True` required whenever `_version` unavailable, including inference tensors and tensor subclasses without readable counter. Unsafe `.data` mutation outside guarantee either way.
- **KL variants:** thirteen teacher-detached variants. `mutual`/`cojeffreys` rejected with `ValueError` even when `kl_weight == 0` — precomputed mode never returns KL gradient to teacher.
- **Teacher CE:** not supported on Triton path.
  - `backend="triton"` + `teacher_ce_weight != 0` → `ValueError`.
  - `backend="auto"` + `teacher_ce_weight != 0` → falls back to Torch reference with one-time fallback warning.
- **Dead rows:** row whose cached logits are all `-inf` treated as zero-support teacher row: teacher CE and KL pinned to `0` (still counts in `mean` denominator).

---

## 3. Aligned-span path (`precomputed_aligned_span`)

`aligned_span_distillation_loss` is a separate objective for cross-tokenizer distillation: instead of per-column KL over shared vocabulary, each aligned span compares the log-likelihood the two models assign to the same text span. Per-span event space is binary — observed continuation versus any other — so the objective is binary KL or binary CE between span log-likelihoods. Intentionally not a `KLVariant`.

**Objectives:** `binary_kl` (default) or `binary_ce`, selected by `AlignedLikelihoodConfig.objective`. Both share the same fused pipeline and one autotune winner.

**Teachers:**

| Teacher | Backend | Notes |
|---|---|---|
| `CrossVocabTeacher(hidden, weight, labels)` | Torch reference only | Live teacher over its own vocabulary; always detached. `backend="triton"` raises `ValueError`; `backend="auto"` uses reference. |
| `PrecomputedSpanTeacher(span_log_probs, *, teacher_ranges=None, trust_teacher_ranges=False)` | Triton (or Torch) | Cached per-span log-likelihoods `[C]`; values must be `<= 0` or `-inf` (`NaN`/`+inf` rejected); must not `require_grad`; same validation-stamp mutation guard as `PrecomputedTeacher`; entry count must equal span count. `teacher_ranges` binds cache to build ranges (checked against `AlignedSpanPairs.teacher_ranges` at call time); raw cache must set it or `trust_teacher_ranges=True`. |

**Span pairs:** `AlignedSpanPairs(student_ranges, teacher_ranges)` — two `(C, 2)` int32/int64 tensors of half-open `[start, end)` token ranges, one teacher range per student range. At call time every range must satisfy `0 <= start < end <= rows` for its side.

**Cases and limits:**

- Span with zero valid (non-ignored) student tokens carries no evidence: excluded from both loss and `mean` denominator.
- With live `CrossVocabTeacher`, all-ignored teacher span likewise dropped. With precomputed teacher only student side gates span — cached `-inf` is distinct, legal zero-support value.
- `build_precomputed_span_teacher` refuses to encode evidence-free span: every teacher span must contain at least one non-ignored teacher token, and error reports offending span indices.
- `KernelConfig` fields for tokenwise KL only (`kl_weight`, `kl_temperature`, `kl_teacher_temperature`, `kl_divergence`, `kl_alpha`, `kl_beta`, `entropy_gate`) rejected with `ValueError` when passed via `kernel_config`.
- Triton pipeline runs two stages: stage A stores per-row NLL and log-sum-exp; FP64 prefix/difference reductions form span losses and one scalar coefficient per token row; stage B converts coefficient plus saved LSE into `a_i * (p - onehot)` logits gradients. One-chunk runs reuse stage-A logits buffer; multi-chunk runs recompute chunk-local logits.
- Student span log-likelihood capped at `-1e-7` in forward objective. Clipped surrogate: backward uses derivative evaluated at cap (straight-through), retained span remains in mean denominator.
- Fixed FP64 workspace is `O(BT + spans)` and not reduced by more chunks. Its OOM reported directly instead of triggering chunk retry loop (only chunk-local vocabulary OOM retryable).
- Path computes no tokenwise KL and no teacher CE; returned `teacher_ce` component always `0`, and `out.kl` holds reduced binary span divergence.

---

## 4. KL divergence variants

`kl_divergence` selects one of fifteen values. Common rules:

- Student distribution is `softmax(student_logits / T_s)`; teacher distribution is `softmax(teacher_logits / T_t)`. Every per-row objective scaled by `T_s**2`.
- **Teacher-detached variants** (all except `mutual`, `cojeffreys`) never send KL gradient to teacher and support asymmetric temperatures (`kl_teacher_temperature != kl_temperature`).
- `mutual` and `cojeffreys` require `SeparateTeacher` or `TiedTeacher`, equal teacher temperature, and route to cotrained paths.
- Because student probabilities come from softmax over finite logits, `p > 0` always holds. Zero teacher support (`q == 0`) can occur through `PrecomputedTeacher` `-inf` logits (e.g. token-map projection). Behavior at zero support defined per variant below.
- `kl_alpha` / `kl_beta` validated whenever provided. When selected variant does not read a parameter, provided value must still be finite float in open interval `(0, 1)`. When KL term inactive, selected variant's own range used for validation; plan canonicalizes value to `0.5` for identity.
- With `kl_weight == 0` no KL computed, variant canonicalizes to `forward`, and both temperatures pinned to `1.0` on plan.

| # | Variant | Objective (per row, before `T_s**2` scale) | `kl_alpha` | `kl_beta` | Teacher grad | Zero-support behavior |
|---|---|---|---|---|---|---|
| 0 | `forward` | `KL(q‖p) = Σ q·(log q − log p)` | — | — | detached | `q == 0` columns contribute exactly `0` |
| 1 | `jeffreys` | `KL(q‖p) + KL(p‖q)` | — | — | detached | reverse half: any `p > 0` where `q == 0` → row `+inf`, zero gradient |
| 2 | `mutual` | `Σ q̄·(log q̄ − log p) + Σ p̄·(log p̄ − log q)` (bars = detached targets) | — | — | **KL gradient** | teacher softmax ⇒ `q > 0`; no special case |
| 3 | `cojeffreys` | symmetric Jeffreys with both sides live | — | — | **KL gradient** | teacher softmax ⇒ `q > 0`; no special case |
| 4 | `reverse` | `KL(p‖q) = Σ p·(log p − log q)` | — | — | detached | `p > 0` where `q == 0` → row `+inf`, zero gradient |
| 5 | `alpha_divergence` | `(1 − Σ p^α q^(1−α)) / (α(1−α))` | `α ∈ (0, 1)`, default `0.5` | — | detached | `q == 0` columns drop out of affinity sum |
| 6 | `skew_forward` | `KL(q ‖ m)`, `m = (1−β)q + βp` | — | `β ∈ (0, 1)`, default `0.5` | detached | finite by construction; bounded by `−log(1−β)` per row |
| 7 | `skew_js` | `(1−β)·KL(q‖m) + β·KL(p‖m)` | — | `β ∈ (0, 1)`, default `0.5` | detached | finite by construction |
| 8 | `entropy_adaptive` | `α(h)·KL(q‖p) + (1−α(h))·KL(p‖q)` with `α(h) = α_min + (α_max−α_min)·σ((h−τ)/scale)`, `h = H(q)/log V` (`0` when `V == 1`) | — (uses `entropy_gate`) | — | detached; gate detached too | reverse half: missing support → row `+inf`, zero gradient |
| 9 | `skew_reverse` | `KL(p ‖ m)`, `m = (1−β)q + βp` | — | `β ∈ (0, 1)`, default `0.5` | detached | finite by construction; bounded by `−log β` per row |
| 10 | `renyi` | teacher-first Rényi `D_r(q‖p) = log(Σ q^r p^(1−r)) / (r−1)` | `r ∈ (0, 2]`, default `0.5`, excluding `\|r−1\| < 0.01` | — | detached | columns with `p = q = 0` skipped; unstable/unsupported row → `+inf` with zero gradient; non-finite non-`-inf` inputs → `NaN` |
| 11 | `renyi_reverse` | student-first Rényi `D_r(p‖q)` | same as `renyi` | — | detached | same sentinels as `renyi` |
| 12 | `dkd` | `λ_t·TCKD + λ_n·NCKD` (target/non-target decoupling) | `λ_t ∈ (0, 16]`, default `1` | `λ_n ∈ (0, 16]`, default `1` | detached | teacher non-target mass over missing student support → row `+inf`, zero KL gradient; CE unchanged |
| 13 | `abkd` | AB-divergence with parameters `(a, b)` | `a ∈ [0.05, 2]`, default `0.5` | `b ∈ [0.05, 2]`, default `0.5` | detached | `p = q = 0` columns contribute `0`; non-finite non-`-inf` inputs → `NaN` with zeroed gradient |
| 14 | `triangular` | triangular discrimination `Σ (p−q)²/(p+q)`, symmetric | — | — | detached | `p + q == 0` columns skipped; bounded by `2` per row |

Variant-specific notes:

- **`mutual` vs `cojeffreys`:** `mutual` follows deep mutual learning — each direction treats the other distribution as a detached target, so the student learns from `KL(q̄‖p)` and the teacher from `KL(p̄‖q)`. `cojeffreys` keeps both halves live for both sides (symmetric Jeffreys co-training objective). Both reject `PrecomputedTeacher`.
- **`renyi` / `renyi_reverse`:** at `r = 1` both limits are ordinary KL — use `forward` or `reverse` there; guard band `|r−1| >= 0.01` enforces this. Near identity (`|r−1| <= 0.125`) the reference switches to cancellation-free `log1p` evaluation with a control variate.
- **`dkd`:** TCKD is binary KL between `(target, non-target)` mass pairs; NCKD is forward KL between non-target distributions. Only variant that reads `labels` inside the KL term, so ignored rows contribute no KL. TCKD remains live at either teacher-target endpoint; NCKD is zero only when the teacher has no non-target support. The DKD paper commonly uses `λ_n ≈ 8`; default here is neutral `(1, 1)` — set `kl_beta=8` for the paper weighting.
- **`entropy_adaptive`:** gate computed from detached teacher distribution and never receives gradient. Defaults: `EntropyGate(alpha_min=0.0, alpha_max=1.0, tau=0.5, scale=0.1)` with constraints `0 <= alpha_min <= alpha_max <= 1`; `scale` and `1 / scale` must both be finite normal float32 values. `entropy_gate` provided with any other variant is accepted (type-checked) but canonicalized away.

### 4.1 Mode × variant compatibility (active KL, `kl_weight != 0`)

| Variant group | `TiedTeacher` | `SeparateTeacher` | `PrecomputedTeacher` |
|---|---|---|---|
| 13 teacher-detached variants | ✓ (frozen/trained paths) | ✓ (frozen/trained paths) | ✓ |
| `mutual`, `cojeffreys` | ✓ (cotrained path) | ✓ (cotrained path) | ✗ `ValueError` (even inactive) |

A CE-trained teacher combined with a detached KL variant receives its CE gradient only.

### 4.2 Temperature limits

- `kl_temperature` and `kl_teacher_temperature` must be finite and `> 0`.
- `1/T_s` and `T_s**2` must be representable as normal float32 values, and `1/T_t` as well — extremely small or large temperatures raise `ValueError` rather than silently flushing to zero/inf in the kernel.
- `kl_teacher_temperature=None` (default) means `T_t = T_s`.
- Both temperatures equal to `1.0` selects a specialized fast path (`t_is_one`) that skips scaling entirely.
- Temperature validation runs even when the KL term is inactive; the plan then pins both temperatures to `1.0`.

---

## 5. Cross-cutting cases and limits

### 5.1 Backend × device × dtype

| Backend | Device | Supported model dtypes | Notes |
|---|---|---|---|
| `torch` | CPU or CUDA/HIP | `float16`, `bfloat16`, `float32`, `float64` | Reference implementation; materializes full `(BT, V)` logits. |
| `triton` | CUDA/HIP only | `float16`, `bfloat16`, `float32` | `bfloat16` requires native hardware support (no emulation); pre-Ampere CUDA rejects it. |
| `auto` | any | union of the above | Uses Triton when available and the policy is supported; otherwise falls back to Torch with one-time process-wide `RuntimeWarning`. |

- All model tensors (`student_hidden`, `weight`, active teacher tensors) must share one dtype outside autocast. Under an active autocast context a supported mixture resolves to the autocast dtype; a bfloat16 autocast on a device without native BF16 falls back to the tensor dtype instead of coercing.
- `PrecomputedTeacher.logits` is exempt from dtype uniformity: cached logits are converted chunk-wise into the compute dtype.
- `float64` is Torch-only. `backend="triton"` with `float64` raises.
- `backend="triton"` raises `RuntimeError` without Triton + CUDA/HIP tensors and `ValueError` for an unsupported dtype. Known unsupported policies are rejected up front with plain `ValueError` naming the fix (precomputed + teacher CE, live `CrossVocabTeacher` on aligned-span path); `UnsupportedTritonPolicyError` (a `ValueError` subclass) is the planning backstop for any remaining semantically valid policy that no Triton specialization executes.
- All public entry points are `torch._dynamo.disable`d: a compiled caller sees a graph break here, and the kernel manages its own fusion.

### 5.2 Labels and `ignore_index`

- `labels` must be `torch.long`, shape `(BT,)`, on the student device.
- Valid values: `[0, V)` or exactly `ignore_index`. Strict host validation is the default and reports up to five offending values.
- `KernelConfig(assume_valid_labels=True)` skips the synchronous host-side range scan — use it when the input pipeline guarantees validity. The Triton kernel keeps its `ignore_index`-safe indexing guard even in trusted mode: an ignored or out-of-range label is clamped to a safe index before any memory access, so trusted mode changes synchronization cost, not memory safety.
- `ignore_index` may be any Python int (default `-100`); it is never used as a memory index.
- Ignored rows are masked before the GEMMs, so non-finite hidden-state values in ignored rows cannot poison the loss or the weight gradients.

### 5.3 Reduction semantics

- `reduction="mean"` divides every component by the number of non-ignored labels, clamped to a minimum of one; `"sum"` returns unnormalized sums.
- Rows pinned to zero (dead precomputed teacher rows, dropped spans) still count in the tokenwise `mean` denominator; dropped spans do not count in the span-mean denominator.

### 5.4 Degenerate inputs

- **`BT == 0`:** returns immediately with zero-valued outputs whose `loss` is still wired to every live gradient source (backward works and produces zero gradients).
- **All weights zero** (`student_ce_weight == teacher_ce_weight == kl_weight == 0`): a differentiable zero loss; nothing is computed.
- **`V` must be `>= 1`**; an empty vocabulary raises.
- **Component reporting:** a component whose weight is `0` is never computed and reads exactly `0` in the output — give a weight a positive value to inspect the component's true value.
- **Output dtype:** `promote_types(input_dtype, float32)` — fp16/bf16 inputs produce float32 loss components; float64 stays float64 (Torch backend only).

### 5.5 Autograd lifecycle

- Each fused Triton forward supports exactly one backward pass. Cached gradient buffers are scaled in place during backward (avoiding full-size copies), which consumes them; a repeated backward raises even when the first call used `retain_graph=True`. Recompute the loss instead.
- The Torch backend has no one-backward restriction. Stabilized `alpha_divergence`, `renyi`, `renyi_reverse`, and `abkd` proxies are first-order only; higher-order requests raise rather than omit KL curvature.
- Inactive objective branches return `None` gradients for inputs that are not part of the graph.

### 5.6 Chunking and memory

- The Triton backend materializes only chunk-local logits; peak vocabulary memory is `O(chunk_rows * V)`. The Torch reference materializes the full `(BT, V)` buffer.
- `ChunkSize.nrows(n)` (fixed rows per chunk, `n >= 1`, clamped to `BT`), `ChunkSize.fast()` (throughput-oriented partition), and `ChunkSize.dynamic()` (workload-shape heuristic) are the three chunk requests. Raw ints, strings, `None`, and legacy sentinels are rejected.
- `num_chunks` is mutually exclusive with `ChunkSize.nrows`, and overrides `ChunkSize.fast()` and `ChunkSize.dynamic()`; the realized non-empty chunk count must not exceed `max_chunks`.
- Dynamic plans start from a pressure heuristic (`(BT/1024)·(V/32768)`, floored by `BT/4096`, rounded to a power of two) bounded by `min(32, 2·max(1, BT//512))` unless `max_chunks` overrides it.
- Fast plans use `ceil(BT/1024)` chunks, at least 2 and at most 16 (unless `max_chunks` overrides the ceiling), ignoring `V`. Past `16·1024` rows the chunks widen instead of multiplying, spending memory to keep launch count low.
- On a retryable CUDA/HIP OOM, dynamic plans free cached allocator blocks, double the chunk count, and retry until `max_chunks`; the final failure message reports the shape and the partition it used, and suggests `num_chunks`. A successful partition is cached (bounded LRU, 1024 entries) keyed by tensor identity (shape/stride/dtype/device/`requires_grad`/contiguity) and plan specialization, so later steps skip failed attempts.
- Fixed plans (`ChunkSize.nrows`/`num_chunks`) and fast plans do not retry. Both automatic modes report the partition they used; below their ceiling they suggest the next partition as `num_chunks`, and at the ceiling they report how much of the chunkable buffer is already gone and what a further split would reclaim. An explicit `num_chunks` suppresses the OOM report.
- `max_fused_size` must be a power of two `<= 32768`; profiles choose `16384` (`balanced`), `32768` (`fast`), `32768` (`debug`).

### 5.7 Autotune interaction

- KL hyperparameters are runtime scalars, so sweeping `kl_beta`, `entropy_gate`, and ordinary `kl_alpha` or temperature values does not re-benchmark. Two boundaries exist: both temperatures being exactly `1.0` toggles `t_is_one`, and active `renyi` / `renyi_reverse` with `|kl_alpha - 1| <= 0.125` selects the `renyi-near-one` pipeline. Crossing either boundary changes autotune identity and may benchmark a separate winner.
- Specialization axes that change autotune identity: kernel path, active objective flags (`compute_kl`, `compute_teacher_ce`), `t_is_one`, the Rényi near-one pipeline, KL style, dtype, and workload shape.
- Aligned-span keeps a distinct pipeline identity; `binary_kl` and `binary_ce` share one winner.

See [reference.md §12](reference.md#12-autotune-environment-variables) for the environment-variable table and cache layout.
