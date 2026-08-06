# orda-kernel

🌐 **Languages:** [English](README.md) | [Tiếng Việt](docs/vi/README.md)

Triton kernels for cross-entropy and KL-divergence knowledge distillation. `orda_kernel` fuses student CE, teacher CE, and the KL term into one chunked kernel that does not materialize the full `(BT, vocab)` logits tensor — the dominant memory cost of distillation on large vocabularies.


**Core capabilities**

- Three teacher modes: tied (shared head), separate (independent head), precomputed (cached)
- 15 KL divergence variants: forward, reverse, Jeffreys, mutual, cojeffreys, alpha-divergence, skew-forward/reverse/JS, entropy-adaptive, Rényi, DKD, ABKD, triangular
- Independent student and teacher softmax temperatures for detached variants

## Documentation

| Document | Contents |
|---|---|
| This README | Installation, quickstart, API surface |
| [docs/guide.md](docs/guide.md) | Task-oriented usage: picking a teacher mode and variant, mixed precision, memory tuning |
| [docs/reference.md](docs/reference.md) | Complete API: functions, parameters, constraints, errors, environment variables |
| [docs/variants.md](docs/variants.md) | Execution variants: kernel paths, KL variants, compatibility, limits |
| [docs/architecture.md](docs/architecture.md) | Execution flow: layout, planning, chunking, autotune |
| [docs/designer.md](docs/designer.md) | Kernel modification guide: contracts, extension checklists |
| [docs/gpu-validation.md](docs/gpu-validation.md) | CUDA/HIP validation commands |
| [docs/ci-setup.md](docs/ci-setup.md) | CI scope and GPU testing |

---

## Installation

Requires `torch>=2.3.0` and Python `>=3.10`.

```bash
pip install orda-kernel
```

**GPU (CUDA):** Add Triton with:

```bash
pip install 'orda-kernel[gpu]'
```

Triton kernels run automatically on CUDA tensors with supported dtypes (`float16`, `float32`, `bfloat16` on native BF16 hardware).

**GPU (ROCm):** Install ROCm Torch first (brings its own Triton), then install ORDA without touching dependencies: `pip install --no-deps orda-kernel`. Runtime detects the Triton distribution automatically. See [docs/gpu-validation.md](docs/gpu-validation.md) for verification.

**CPU:** No Triton path exists. Calls run the Torch implementation, which materializes full logits — usable for development and testing, not for the memory profile the kernels exist to provide.

```python
import orda_kernel
orda_kernel.is_available()  # True when Triton + CUDA/HIP kernels can run
```

Optional extras: `test` (pytest), `ci` (ruff, mypy, pytest plugins), `gpu` (triton). Wheels include PEP 561 `py.typed` marker.

**Install from source:**

```bash
git clone https://github.com/hiwuhgds-pixel/ORDA-Knowledge-Distillation-Kernel orda-kernel
cd orda-kernel
pip install .
```

---

## Quickstart

Inputs:

- `student_hidden`: `(BT, hidden)` — flattened batch × seq hidden states
- `weight`: `(vocab, hidden)` — student output projection
- `labels`: `(BT,)`, `torch.long` — targets, with `ignore_index` for masked positions
- `teacher`: one of three teacher descriptors

### Functional API

```python
import torch
from orda_kernel import distillation_loss, TiedTeacher

bt, hidden, vocab = 4096, 2048, 32000
student_hidden = torch.randn(bt, hidden, device="cuda", requires_grad=True)
weight = torch.randn(vocab, hidden, device="cuda", requires_grad=True)
labels = torch.randint(0, vocab, (bt,), device="cuda")

teacher_hidden = torch.randn(bt, hidden, device="cuda")

out = distillation_loss(
    student_hidden,
    weight,
    labels,
    TiedTeacher(hidden=teacher_hidden),
    student_ce_weight=1.0,
    kl_weight=1.0,
    kl_temperature=2.0,
    kl_divergence="forward",
)

out.loss.backward()
print(out.loss, out.student_ce, out.kl)
```

### Module API

```python
from orda_kernel import DistillationLoss, SeparateTeacher

criterion = DistillationLoss(
    student_ce_weight=1.0,
    teacher_ce_weight=0.0,
    kl_weight=1.0,
    kl_temperature=2.0,
    kl_divergence="jeffreys",
    reduction="mean",
)

out = criterion(
    student_hidden,
    weight,
    labels,
    SeparateTeacher(hidden=teacher_hidden, weight=teacher_weight),
)
out.loss.backward()
```

`distillation_loss` returns `DistillationLossOutput(loss, student_ce, teacher_ce, kl)`. Components are unweighted for logging. Each fused Triton forward supports one backward pass.

### Teacher descriptors

| Descriptor | Fields | Meaning |
|---|---|---|
| `TiedTeacher(hidden, mode=None)` | `hidden: (BT, hidden)` | Student and teacher share projection `weight`. Optional `mode` (`"trained"`/`"frozen"`/`"cotrained"`) asserts role without changing routing. |
| `SeparateTeacher(hidden, weight)` | `hidden: (BT, D_t)`, `weight: (vocab, D_t)` | Teacher has independent hidden states and projection. Gradients flow to tensors with `requires_grad=True`. |
| `PrecomputedTeacher(logits=… \| teacher_hidden=…, teacher_weight=…)` | `logits: (BT, vocab)` or `teacher_hidden: (BT, D_t)` + `teacher_weight: (vocab, D_t)` | Cached teacher outputs. No gradient reaches teacher; cached tensors must not `require_grad`. In-place mutation after construction is detected. |

Teacher role derives from weights: `kl_divergence ∈ {"mutual","cojeffreys"}` with `kl_weight > 0` co-trains teacher; otherwise `teacher_ce_weight > 0` trains via CE; otherwise frozen. `teacher_ce_weight=None` defaults to `1.0` for tied teachers and `0.0` for separate/precomputed. See [docs/guide.md](docs/guide.md#1-choosing-a-teacher-mode) for how to pick one, and [docs/variants.md §1](docs/variants.md#1-how-a-call-resolves-to-a-variant) for routing tables.

---

## Parameters

`distillation_loss(student_hidden, weight, labels, teacher, *, ...)`

See [docs/reference.md §1](docs/reference.md#1-distillation_loss) for complete constraints.

| Keyword | Default | Description |
|---|---|---|
| `student_ce_weight` / `teacher_ce_weight` / `kl_weight` | `1.0` / `None` / `0.0` | Term weights (finite, `>= 0`). |
| `kl_temperature` | `1.0` | Student softmax temperature; each KL row scales by `T_s**2`. |
| `kl_teacher_temperature` | `None` | Teacher temperature (defaults to `T_s`). Allowed for teacher-detached variants; `mutual`/`cojeffreys` require `T_t == T_s`. |
| `kl_divergence` | `"forward"` | One of 15 variants. |
| `kl_alpha`, `kl_beta`, `entropy_gate` | `None` | Variant hyperparameters with per-variant ranges. |
| `ignore_index` | `-100` | Label value excluded from CE/KL; never used as memory index. |
| `reduction` | `"mean"` | `"mean"` or `"sum"`. |
| `profile` | `"balanced"` | `"balanced"`, `"fast"`, `"debug"`. |
| `backend` | `"auto"` | `"auto"`, `"triton"`, `"torch"`. |
| `config` | `None` | `KernelConfig` overrides: chunking, autotune mode, weight-gradient accumulation, `assume_valid_labels`. |
| `gradient_materialization_scale` / `grad_scaler` | `None` / `None` | Precision control for fused-path gradient storage under fp16 — see [docs/guide.md §4](docs/guide.md#4-mixed-precision-and-gradient-scaling). |

## KL variants

`p` = student, `q` = teacher. All variants except `mutual`/`cojeffreys` detach teacher.

See [docs/variants.md §4](docs/variants.md#4-kl-divergence-variants) for formulas, ranges, zero-support behavior.

| Variant | Objective | Parameters |
|---|---|---|
| `forward` | `KL(q‖p)` | — |
| `reverse` | `KL(p‖q)` | — |
| `jeffreys` | `KL(q‖p) + KL(p‖q)` | — |
| `mutual` | mutual-learning KL (teacher receives gradient) | — |
| `cojeffreys` | symmetric co-training KL (teacher receives gradient) | — |
| `alpha_divergence` | power α-divergence | `kl_alpha ∈ (0,1)` |
| `skew_forward` | `KL(q‖(1−β)q+βp)` | `kl_beta ∈ (0,1)` |
| `skew_reverse` | `KL(p‖(1−β)q+βp)` | `kl_beta ∈ (0,1)` |
| `skew_js` | skew Jensen–Shannon | `kl_beta ∈ (0,1)` |
| `entropy_adaptive` | teacher-entropy-gated forward/reverse blend | `entropy_gate` |
| `renyi` / `renyi_reverse` | Rényi `D_r(q‖p)` / `D_r(p‖q)` | `kl_alpha = r ∈ (0,2]`, `\|r−1\| ≥ 0.01` |
| `dkd` | `λ_t·TCKD + λ_n·NCKD` | `kl_alpha=λ_t`, `kl_beta=λ_n ∈ (0,16]` |
| `abkd` | AB-divergence | `kl_alpha`, `kl_beta ∈ [0.05,2]` |
| `triangular` | triangular discrimination | — |

`mutual` and `cojeffreys` require `TiedTeacher` or `SeparateTeacher` and route to cotrained kernel paths. With `TiedTeacher`, the shared projection weight accumulates gradient from both student and teacher halves.

---

## Cross-vocabulary distillation

When student and teacher use different tokenizers, per-column KL is undefined. Three tools handle this:

**Token-map projection → `PrecomputedTeacher`**

`build_exact_token_map(student_vocab, teacher_vocab)` + `project_teacher_logits(...)` project teacher distribution onto matching student tokens. Duplicates sum, unmatched columns get `-inf`. Pass `temperature=T` to temper before aggregation and declare with `PrecomputedTeacher(logits=..., logits_temperature=T)`.

**Multi-teacher mixing → `PrecomputedTeacher`**

`mix_teacher_logits([...], weights, mode="arithmetic"|"geometric")` mixes cached teacher distributions (probability average or product-of-experts). `generalized_jensen_shannon` provides per-row agreement diagnostic.

**Span-aligned likelihoods**

`aligned_span_distillation_loss` compares log-likelihood each model assigns to text spans:

```python
from orda_kernel import (
    AlignedLikelihoodConfig, AlignedSpanPairs,
    aligned_span_distillation_loss, build_precomputed_span_teacher,
)

pairs = AlignedSpanPairs(student_ranges, teacher_ranges)  # int [C, 2] half-open

span_teacher = build_precomputed_span_teacher(
    teacher_hidden, teacher_weight, teacher_labels, pairs.teacher_ranges,
)
out = aligned_span_distillation_loss(
    student_hidden, student_weight, student_labels,
    span_teacher, pairs,
    AlignedLikelihoodConfig(span_kl_weight=0.7),
)
out.loss.backward()
```

Spans with no valid tokens are dropped. Alignment (tokenizer offsets, DP matching) happens outside the package.

See [docs/guide.md §6](docs/guide.md#6-cross-vocabulary-distillation) for choosing among the three.

---

## Backends

| `backend` | Behavior |
|---|---|
| `"auto"` (default) | Fused Triton kernel when Triton, a CUDA/HIP device, a supported dtype, and a supported policy are all present; otherwise the Torch implementation with a one-time warning naming the reason. |
| `"triton"` | Requires the fused path; raises instead of falling back. |
| `"torch"` | Always the reference implementation; materializes full `(BT, vocab)` logits. |

Dtype support: Triton `{fp16, bf16, fp32}` (bf16 needs native hardware); Torch adds `fp64`.

---

## Testing

```bash
pip install '.[test]'
python test/run_test_cpu.py
```

CPU runner executes unit, CPU-reference, and multiprocess tests. GPU correctness and autotune suites are owned by `test/run_test_gpu.py`:

```bash
python test/run_test_gpu.py --test public-api
python test/run_test_gpu.py --route tied --kl forward
```

See [docs/gpu-validation.md](docs/gpu-validation.md) for the full GPU gate.

Lint and type-check:

```bash
pip install '.[ci]'
python -m ruff format --check src test scripts .github/scripts benchmark
python -m ruff check src test scripts .github/scripts benchmark
python -m mypy
```
