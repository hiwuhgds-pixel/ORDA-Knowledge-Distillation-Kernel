# Usage guide

Task-oriented patterns for common distillation scenarios. For the complete parameter contract, see [reference.md](reference.md); for kernel routing tables and variant details, see [variants.md](variants.md).

---

## 1. Choosing a teacher mode

**Use `TiedTeacher` when:**
- Student and teacher share one projection weight
- EMA self-distillation (frozen teacher from student's own EMA)
- Mutual learning with co-training KL variants (`mutual`, `cojeffreys`)
- You want one GEMM instead of two

**Use `SeparateTeacher` when:**
- Teacher has independent projection weight
- Distilling from a larger model with different hidden dimension
- Co-training both models with `mutual` or `cojeffreys` KL
- Teacher gradients are needed

**Use `PrecomputedTeacher` when:**
- Teacher outputs are cached from an earlier pass
- Teacher has no learnable parameters (frozen external model)
- Student and teacher use different tokenizers (after projection with `project_teacher_logits`)
- You're mixing multiple teachers with `mix_teacher_logits`

### Example: EMA self-distillation

```python
import copy
import torch
from orda_kernel import DistillationLoss, TiedTeacher

ema_trunk = copy.deepcopy(student_trunk).requires_grad_(False)
criterion = DistillationLoss(
    teacher_ce_weight=0.0,  # frozen teacher
    kl_weight=1.0,
    kl_temperature=2.0,
    kl_divergence="skew_forward",
)

for batch in loader:
    student_hidden = student_trunk(batch.tokens)
    with torch.no_grad():
        teacher_hidden = ema_trunk(batch.tokens)

    out = criterion(
        student_hidden.flatten(0, 1),
        head_weight,
        batch.labels.flatten(),
        TiedTeacher(teacher_hidden.flatten(0, 1), mode="frozen"),
    )
    out.loss.backward()
    optimizer.step(); optimizer.zero_grad()

    # EMA update
    with torch.no_grad():
        for p_ema, p in zip(ema_trunk.parameters(), student_trunk.parameters()):
            p_ema.lerp_(p, 1.0 - 0.999)
```

The `mode="frozen"` assertion is optional — the kernel path is selected from `teacher_ce_weight=0.0` and the absence of co-training KL variants. The assertion documents intent and raises if the routing disagrees.

### Example: Separate teacher with gradient

```python
from orda_kernel import distillation_loss, SeparateTeacher

student_trunk = ...  # hidden_dim=768
teacher_trunk = ...  # hidden_dim=1024, requires_grad=True

student_hidden = student_trunk(tokens)
teacher_hidden = teacher_trunk(tokens)

out = distillation_loss(
    student_hidden.flatten(0, 1),
    student_head_weight,  # (vocab, 768)
    labels.flatten(),
    SeparateTeacher(
        hidden=teacher_hidden.flatten(0, 1),
        weight=teacher_head_weight,  # (vocab, 1024)
    ),
    teacher_ce_weight=1.0,  # teacher receives CE gradient
    kl_weight=1.0,
)
out.loss.backward()
```

Gradients flow to both `teacher_hidden` and `teacher_head_weight` when `requires_grad=True`.

### Example: Precomputed (cached) teacher

```python
from orda_kernel import distillation_loss, PrecomputedTeacher

# Cache teacher outputs once
with torch.no_grad():
    teacher_logits = teacher_trunk(tokens) @ teacher_head_weight.T

# Reuse across student training steps
teacher = PrecomputedTeacher(logits=teacher_logits)

for step in range(num_steps):
    student_hidden = student_trunk(tokens)
    out = distillation_loss(
        student_hidden.flatten(0, 1),
        student_head_weight,
        labels.flatten(),
        teacher,
        teacher_ce_weight=0.0,  # no teacher gradient
        kl_weight=1.0,
    )
    out.loss.backward()
```

Cached tensors must not `require_grad`. In-place mutation after construction raises `ValueError`.

---

## 2. Choosing a KL variant

**Forward KL** (`"forward"`): classic teacher-to-student knowledge transfer. Teacher assigns high probability → student learns it.

**Reverse KL** (`"reverse"`): mode-seeking. Student concentrates mass where teacher has support. Drops teacher's low-probability tail.

**Mutual/Cojeffreys** (`"mutual"`, `"cojeffreys"`): co-training. Teacher receives gradient and learns from student. Requires `TiedTeacher` or `SeparateTeacher`.

**Skew variants** (`"skew_forward"`, `"skew_js"`, `"skew_reverse"`): interpolates teacher and student with `kl_beta`. Lower `beta` = more conservative.

**Entropy-adaptive** (`"entropy_adaptive"`): blends forward/reverse based on teacher entropy. High-confidence teacher rows use forward KL; low-confidence rows use reverse.

**Alpha-divergence** (`"alpha_divergence"`): power-mean interpolation controlled by `kl_alpha ∈ (0,1)`. Near 0 → reverse-like; near 1 → forward-like.

**Rényi** (`"renyi"`, `"renyi_reverse"`): generalizes KL with order parameter `kl_alpha=r ∈ (0,2]`, excluding `|r−1| < 0.01`.

**DKD** (`"dkd"`): decoupled knowledge distillation = `λ_t·TCKD + λ_n·NCKD`. Target-class and non-target components weighted by `kl_alpha=λ_t`, `kl_beta=λ_n`.

**ABKD** (`"abkd"`): AB-divergence with `kl_alpha`, `kl_beta ∈ [0.05, 2]`.

See [variants.md §4](variants.md#4-kl-divergence-variants) for formulas and numeric ranges.

---

## 3. Temperature and reduction

**Student temperature** (`kl_temperature`): softens student logits before KL. Each row scales by `T_s**2`. Higher temperature → smoother gradients.

**Teacher temperature** (`kl_teacher_temperature`): defaults to `kl_temperature`. Allowed for detached variants; `mutual`/`cojeffreys` require `T_t == T_s`.

**Reduction** (`"mean"` or `"sum"`): `"mean"` divides by non-ignored token count (clamped `>= 1`). `"sum"` returns the batch total.

### Pattern: high temperature for early training

```python
for epoch in range(num_epochs):
    temp = 4.0 if epoch < warmup_epochs else 2.0
    criterion.kl_temperature = temp
    # ... training loop
```

---

## 4. Mixed precision and gradient scaling

The fused kernel writes logit gradients to an fp16 buffer during `forward()`, before autograd runs. A `GradScaler` scales the loss, so its factor only arrives in `backward()` — after those values are stored, and after anything below the subnormal floor has flushed to zero. Multiplying a stored zero back up recovers nothing, so the fused path needs a factor applied *inside* the kernel, before the store. That is what `gradient_materialization_scale` does; see [reference.md — Gradient scaling under fp16](reference.md#gradient-scaling-under-fp16) for the full mechanism.

Pass `grad_scaler=` and the kernel takes that factor from your existing scaler.

**Without AMP:**
```python
out = distillation_loss(student_hidden, weight, labels, teacher, kl_weight=1.0)
out.loss.backward()
```

**With `torch.amp.GradScaler`:**
```python
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast("cuda", dtype=torch.float16):
    out = distillation_loss(
        student_hidden, weight, labels, teacher,
        kl_weight=1.0,
        grad_scaler=scaler,  # kernel applies scale internally
    )

scaler.scale(out.loss).backward()
scaler.step(optimizer)
scaler.update()
```

Keep the usual `scaler.scale(loss).backward()` call: the kernel cancels its own factor in backward, so `scaler.step()` sees exactly the gradient it expects.

**Equivalently, with explicit scale:**
```python
scale = torch.full((), 1024.0, dtype=torch.float32, device="cuda")

with torch.amp.autocast("cuda", dtype=torch.float16):
    out = distillation_loss(
        student_hidden, weight, labels, teacher,
        kl_weight=1.0,
        gradient_materialization_scale=scale,
    )

(out.loss * scale).backward()
# Manually unscale gradients before optimizer.step()
for p in [student_hidden, weight]:
    if p.grad is not None:
        p.grad.div_(scale)
```

**Torch backend:** neither argument has an effect — that path uses autograd directly and never materializes a low-precision gradient buffer.

**`grad_scaler` and `gradient_materialization_scale` are mutually exclusive.** They set the same factor; passing both raises `ValueError`.

---

## 5. Memory tuning

The fused kernel splits the batch into row chunks and materializes only chunk-local logits. Default chunk policy is `ChunkSize.dynamic()`, which picks partition size from workload shape.

**Profile defaults:**

| Profile | Chunk policy | Autotune | Weight-grad accumulation |
|---|---|---|---|
| `balanced` (default) | `dynamic()` | `True` | `auto` |
| `fast` | `ChunkSize.fast()` | `"Max"` | `auto` |
| `debug` | `dynamic()` | `False` | `fp32` |

### Forcing a fixed partition

```python
from orda_kernel import distillation_loss, KernelConfig, ChunkSize

out = distillation_loss(
    student_hidden, weight, labels, teacher,
    kl_weight=1.0,
    config=KernelConfig(chunk_size=ChunkSize.nrows(512)),
)
```

### Requesting a chunk count

```python
config = KernelConfig(num_chunks=8)
```

`num_chunks` overrides `ChunkSize.fast()`, `ChunkSize.dynamic()`, and `max_chunks`; the kernel computes `chunk_size = ceil(BT / num_chunks)`. It is mutually exclusive with `ChunkSize.nrows(n)` — passing both raises `ValueError`.

### OOM recovery

Dynamic plans retry with doubled chunk count after recoverable OOM and cache the working partition. The cache is keyed by device, tensor shapes, dtype, and specialization, and is independent of the autotune cache. `ChunkSize.nrows(n)` and an explicit `num_chunks` are fixed partitions: they raise on OOM instead of retrying.

**Fast profile:** uses `ChunkSize.fast()` (1024 rows/chunk, 2–16 chunks). Never retries OOM — raises with the partition it used and a suggested `num_chunks`.

---

## 6. Cross-vocabulary distillation

When student and teacher use different tokenizers, three tools are available:

### 6.1 Token-map projection

Project teacher logits onto student vocabulary with exact string match:

```python
from orda_kernel import (
    build_exact_token_map, project_teacher_logits, PrecomputedTeacher
)

# Build map once
token_map = build_exact_token_map(student_vocab, teacher_vocab)  # [V_t] int64

# Project teacher logits for each batch
with torch.no_grad():
    teacher_logits = teacher_trunk(tokens) @ teacher_weight.T
    projected = project_teacher_logits(
        teacher_logits, token_map, student_vocab_size,
        unmatched="renormalize",  # or "mask"
        temperature=2.0,  # temper before aggregation
    )

teacher = PrecomputedTeacher(logits=projected, logits_temperature=2.0)
```

Mass from duplicate teacher tokens is summed; unmatched student columns become `-inf` (zero support). `unmatched="renormalize"` rescales matched mass to one; `unmatched="mask"` keeps the absolute matched log-mass.

### 6.2 Multi-teacher mixing

Mix cached teacher distributions:

```python
from orda_kernel import mix_teacher_logits

teacher_logits_list = [teacher_a_logits, teacher_b_logits]
weights = [0.7, 0.3]

mixed = mix_teacher_logits(
    teacher_logits_list, weights,
    mode="arithmetic",  # or "geometric"
)
teacher = PrecomputedTeacher(logits=mixed)
```

`mode="arithmetic"` averages probabilities; `mode="geometric"` is product-of-experts.

### 6.3 Span-aligned likelihoods

Compare log-likelihood each model assigns to text spans:

```python
from orda_kernel import (
    AlignedSpanPairs, AlignedLikelihoodConfig,
    aligned_span_distillation_loss, build_precomputed_span_teacher,
)

# Alignment from external tokenizer offset matching
student_ranges = torch.tensor([[0, 3], [3, 7]], dtype=torch.int64)  # [C, 2]
teacher_ranges = torch.tensor([[0, 4], [4, 9]], dtype=torch.int64)

pairs = AlignedSpanPairs(student_ranges, teacher_ranges)

# Cache teacher span log-probs
span_teacher = build_precomputed_span_teacher(
    teacher_hidden, teacher_weight, teacher_labels, pairs.teacher_ranges,
)

# Distill
out = aligned_span_distillation_loss(
    student_hidden, student_weight, student_labels,
    span_teacher, pairs,
    AlignedLikelihoodConfig(
        student_ce_weight=1.0,
        span_kl_weight=0.7,
        objective="binary_kl",  # or "binary_ce"
    ),
)
```

Spans with no valid tokens are dropped. Alignment (DP matching, byte-offset mapping) happens outside the package.

**When to use each:**

- **Token-map:** fast, exact-match only, loses unmatched teacher knowledge
- **Multi-teacher:** ensemble multiple cached teachers, no vocabulary constraint
- **Span-aligned:** handles arbitrary tokenizer differences, requires span alignment preprocessing

---

## 7. Autotune

First execution benchmarks launch configurations and caches the winner in-memory and on disk (SQLite). Cache key: device fingerprint, workload shape, dtype, pipeline identity, and a timed-source AST digest. KL hyperparameters and temperatures are runtime scalars — sweeping them never re-benchmarks.

**Cache location:** `ORDA_AUTOTUNE_CACHE_DIR`, else platform cache dir (`~/.cache/orda_kernel` on Linux, `%LOCALAPPDATA%/orda_kernel` on Windows).

**On clusters:** point `ORDA_AUTOTUNE_CACHE_DIR` to node-local storage. SQLite WAL + OS locking are not reliable on every NFS.

**Autotune modes:**

| Mode | Behavior |
|---|---|
| `autotune=False` | Use default config; no benchmarking |
| `autotune=True` | Screen candidates, advance finalists to robust timing |
| `autotune="Max"` | Deeper measurement with fixed warmup/rep controls |

`refresh_autotune=True` re-benchmarks once per process and workload, honoring the effective `autotune` mode's depth.

**Disabling autotune:**

```python
config = KernelConfig(autotune=False)
```

**Clearing cache:**

```python
import orda_kernel
orda_kernel.reset_autotune_cache()
```

See [reference.md §12](reference.md#12-autotune-environment-variables) for all environment knobs.

---

## 8. Common mistakes

**1. Forgetting that a tied teacher trains by default**

```python
# teacher_ce_weight defaults to 1.0 here — the teacher half receives CE gradient
out = distillation_loss(
    student_hidden, weight, labels,
    TiedTeacher(teacher_hidden),
    kl_weight=1.0,
)
```

For tied teachers `teacher_ce_weight=None` resolves to `1.0` (trained path); for separate and precomputed it resolves to `0.0`. Pass `teacher_ce_weight=0.0` for a frozen tied teacher, and add `mode="frozen"` if you want that intent checked.

**2. Forgetting to detach cached teacher tensors**

```python
# Wrong: raises "cached tensors must not require_grad"
teacher = PrecomputedTeacher(logits=teacher_logits)  # teacher_logits.requires_grad=True
```

Use `with torch.no_grad()` or `.detach()` before constructing `PrecomputedTeacher`.

**3. Reusing a precomputed teacher after in-place mutation**

```python
teacher = PrecomputedTeacher(logits=cached_logits)
cached_logits.add_(1.0)  # ← in-place mutation
out = distillation_loss(..., teacher)  # raises ValueError
```

Construct a new descriptor instead.

**4. Expecting `GradScaler` alone to prevent fp16 underflow on the fused path**

```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    out = distillation_loss(..., kl_weight=1.0)  # ← no grad_scaler=
scaler.scale(out.loss).backward()
```

The loss scale reaches `backward()`, but this kernel already stored its gradients to fp16 during `forward()`; underflowed values are zero by then. Pass `grad_scaler=scaler` so the factor is applied before the store — see [section 4](#4-mixed-precision-and-gradient-scaling).

**5. Passing both `gradient_materialization_scale` and `grad_scaler`**

```python
# Wrong: mutually exclusive
out = distillation_loss(
    ...,
    gradient_materialization_scale=scale,
    grad_scaler=scaler,  # ← raises ValueError
)
```

Pick one or neither.
