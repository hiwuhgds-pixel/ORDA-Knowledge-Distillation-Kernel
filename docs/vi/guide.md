# Hướng dẫn sử dụng

Các mẫu theo tác vụ cho các tình huống distillation phổ biến. Đối với hợp đồng tham số đầy đủ, xem [reference.md](reference.md); đối với bảng routing kernel và chi tiết variant, xem [variants.md](variants.md).

---

## 1. Chọn teacher mode

**Dùng `TiedTeacher` khi:**
- Student và teacher dùng chung một projection weight
- EMA self-distillation (frozen teacher từ EMA của student)
- Mutual learning với co-training KL variants (`mutual`, `cojeffreys`)
- Bạn muốn một GEMM thay vì hai

**Dùng `SeparateTeacher` khi:**
- Teacher có projection weight độc lập
- Chưng cất từ model lớn hơn với hidden dimension khác
- Co-training cả hai model với `mutual` hoặc `cojeffreys` KL
- Cần teacher gradients

**Dùng `PrecomputedTeacher` khi:**
- Teacher outputs được cache từ pass trước đó
- Teacher không có learnable parameters (frozen external model)
- Student và teacher dùng tokenizer khác nhau (sau projection với `project_teacher_logits`)
- Bạn đang trộn nhiều teacher với `mix_teacher_logits`

### Ví dụ: EMA self-distillation

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

`mode="frozen"` assertion là tùy chọn — kernel path được chọn từ `teacher_ce_weight=0.0` và không có co-training KL variants. Assertion ghi lại ý định và raise nếu routing không khớp.

### Ví dụ: Separate teacher với gradient

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
    teacher_ce_weight=1.0,  # teacher nhận CE gradient
    kl_weight=1.0,
)
out.loss.backward()
```

Gradients chảy đến cả `teacher_hidden` và `teacher_head_weight` khi `requires_grad=True`.

### Ví dụ: Precomputed (cached) teacher

```python
from orda_kernel import distillation_loss, PrecomputedTeacher

# Cache teacher outputs một lần
with torch.no_grad():
    teacher_logits = teacher_trunk(tokens) @ teacher_head_weight.T

# Tái sử dụng qua các student training steps
teacher = PrecomputedTeacher(logits=teacher_logits)

for step in range(num_steps):
    student_hidden = student_trunk(tokens)
    out = distillation_loss(
        student_hidden.flatten(0, 1),
        student_head_weight,
        labels.flatten(),
        teacher,
        teacher_ce_weight=0.0,  # không có teacher gradient
        kl_weight=1.0,
    )
    out.loss.backward()
```

Cached tensors phải không `require_grad`. In-place mutation sau construction raise `ValueError`.

---

## 2. Chọn KL variant

**Forward KL** (`"forward"`): truyền kiến thức teacher-to-student cổ điển. Teacher gán xác suất cao → student học nó.

**Reverse KL** (`"reverse"`): mode-seeking. Student tập trung mass nơi teacher có support. Bỏ đuôi xác suất thấp của teacher.

**Mutual/Cojeffreys** (`"mutual"`, `"cojeffreys"`): co-training. Teacher nhận gradient và học từ student. Yêu cầu `TiedTeacher` hoặc `SeparateTeacher`.

**Skew variants** (`"skew_forward"`, `"skew_js"`, `"skew_reverse"`): nội suy teacher và student với `kl_beta`. `beta` thấp hơn = bảo thủ hơn.

**Entropy-adaptive** (`"entropy_adaptive"`): pha trộn forward/reverse dựa trên teacher entropy. Teacher rows độ tin cậy cao dùng forward KL; rows độ tin cậy thấp dùng reverse.

**Alpha-divergence** (`"alpha_divergence"`): nội suy power-mean điều khiển bởi `kl_alpha ∈ (0,1)`. Gần 0 → reverse-like; gần 1 → forward-like.

**Rényi** (`"renyi"`, `"renyi_reverse"`): tổng quát hóa KL với order parameter `kl_alpha=r ∈ (0,2]`, trừ `|r−1| < 0.01`.

**DKD** (`"dkd"`): decoupled knowledge distillation = `λ_t·TCKD + λ_n·NCKD`. Target-class và non-target components trọng số bởi `kl_alpha=λ_t`, `kl_beta=λ_n`.

**ABKD** (`"abkd"`): AB-divergence với `kl_alpha`, `kl_beta ∈ [0.05, 2]`.

Xem [variants.md §4](variants.md#4-các-biến-thể-kl-divergence) cho công thức và phạm vi số học.

---

## 3. Temperature và reduction

**Student temperature** (`kl_temperature`): làm mềm student logits trước KL. Mỗi row scale bởi `T_s**2`. Nhiệt độ cao hơn → gradients mượt hơn.

**Teacher temperature** (`kl_teacher_temperature`): mặc định `kl_temperature`. Cho phép với detached variants; `mutual`/`cojeffreys` yêu cầu `T_t == T_s`.

**Reduction** (`"mean"` hoặc `"sum"`): `"mean"` chia cho non-ignored token count (clamped `>= 1`). `"sum"` trả về tổng batch.

### Mẫu: nhiệt độ cao cho đầu training

```python
for epoch in range(num_epochs):
    temp = 4.0 if epoch < warmup_epochs else 2.0
    criterion.kl_temperature = temp
    # ... training loop
```

---

## 4. Mixed precision và gradient scaling

Fused kernel ghi logit gradients xuống buffer fp16 ngay trong `forward()`, trước khi autograd chạy. `GradScaler` scale loss, nên hệ số của nó chỉ tới ở `backward()` — sau khi các giá trị đó đã được lưu, và sau khi mọi thứ dưới ngưỡng subnormal đã bị đẩy về zero. Nhân ngược một zero đã lưu thì không khôi phục được gì, nên đường fused cần một hệ số áp *bên trong* kernel, trước lần ghi. Đó chính là việc của `gradient_materialization_scale`; xem [reference.md — Gradient scaling dưới fp16](reference.md#gradient-scaling-dưới-fp16) cho cơ chế đầy đủ.

Truyền `grad_scaler=` và kernel sẽ lấy hệ số đó từ scaler sẵn có của bạn.

**Không có AMP:**
```python
out = distillation_loss(student_hidden, weight, labels, teacher, kl_weight=1.0)
out.loss.backward()
```

**Với `torch.amp.GradScaler`:**
```python
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast("cuda", dtype=torch.float16):
    out = distillation_loss(
        student_hidden, weight, labels, teacher,
        kl_weight=1.0,
        grad_scaler=scaler,  # kernel áp scale bên trong
    )

scaler.scale(out.loss).backward()
scaler.step(optimizer)
scaler.update()
```

Vẫn giữ lời gọi `scaler.scale(loss).backward()` như thường: kernel tự khử hệ số của nó ở backward, nên `scaler.step()` nhận đúng gradient mà nó mong đợi.

**Tương đương, với explicit scale:**
```python
scale = torch.full((), 1024.0, dtype=torch.float32, device="cuda")

with torch.amp.autocast("cuda", dtype=torch.float16):
    out = distillation_loss(
        student_hidden, weight, labels, teacher,
        kl_weight=1.0,
        gradient_materialization_scale=scale,
    )

(out.loss * scale).backward()
# Unscale gradients thủ công trước optimizer.step()
for p in [student_hidden, weight]:
    if p.grad is not None:
        p.grad.div_(scale)
```

**Torch backend:** cả hai tham số đều không có hiệu lực — đường đó dùng autograd trực tiếp và không bao giờ tạo low-precision gradient buffer.

**`grad_scaler` và `gradient_materialization_scale` loại trừ lẫn nhau.** Chúng đặt cùng một hệ số; truyền cả hai raise `ValueError`.

---

## 5. Tinh chỉnh bộ nhớ

Fused kernel chia batch thành row chunks và chỉ tạo chunk-local logits. Chunk policy mặc định là `ChunkSize.dynamic()`, chọn kích thước partition từ workload shape.

**Profile mặc định:**

| Profile | Chunk policy | Autotune | Weight-grad accumulation |
|---|---|---|---|
| `balanced` (mặc định) | `dynamic()` | `True` | `auto` |
| `fast` | `ChunkSize.fast()` | `"Max"` | `auto` |
| `debug` | `dynamic()` | `False` | `fp32` |

### Ép partition cố định

```python
from orda_kernel import distillation_loss, KernelConfig, ChunkSize

out = distillation_loss(
    student_hidden, weight, labels, teacher,
    kl_weight=1.0,
    config=KernelConfig(chunk_size=ChunkSize.nrows(512)),
)
```

### Yêu cầu chunk count

```python
config = KernelConfig(num_chunks=8)
```

`num_chunks` override `ChunkSize.fast()`, `ChunkSize.dynamic()` và `max_chunks`; kernel tính `chunk_size = ceil(BT / num_chunks)`. Nó loại trừ lẫn nhau với `ChunkSize.nrows(n)` — truyền cả hai sẽ raise `ValueError`.

### OOM recovery

Dynamic plans retry với doubled chunk count sau recoverable OOM và cache working partition. Cache được key bởi device, tensor shapes, dtype, và specialization, và độc lập với cache autotune. `ChunkSize.nrows(n)` và `num_chunks` tường minh là các partition cố định: chúng raise khi OOM thay vì retry.

**Fast profile:** dùng `ChunkSize.fast()` (1024 rows/chunk, 2–16 chunks). Không bao giờ retry OOM — raise kèm phân hoạch nó đã dùng và một `num_chunks` được đề xuất.

---

## 6. Chưng cất chéo từ vựng

Khi student và teacher dùng tokenizer khác nhau, ba công cụ có sẵn:

### 6.1 Token-map projection

Chiếu teacher logits lên student vocabulary với exact string match:

```python
from orda_kernel import (
    build_exact_token_map, project_teacher_logits, PrecomputedTeacher
)

# Build map một lần
token_map = build_exact_token_map(student_vocab, teacher_vocab)  # [V_t] int64

# Chiếu teacher logits cho mỗi batch
with torch.no_grad():
    teacher_logits = teacher_trunk(tokens) @ teacher_weight.T
    projected = project_teacher_logits(
        teacher_logits, token_map, student_vocab_size,
        unmatched="renormalize",  # hoặc "mask"
        temperature=2.0,  # temper trước aggregation
    )

teacher = PrecomputedTeacher(logits=projected, logits_temperature=2.0)
```

Mass từ duplicate teacher tokens được sum; unmatched student columns trở thành `-inf` (zero support). `unmatched="renormalize"` rescale matched mass về một; `unmatched="mask"` giữ matched log-mass tuyệt đối.

### 6.2 Multi-teacher mixing

Trộn cached teacher distributions:

```python
from orda_kernel import mix_teacher_logits

teacher_logits_list = [teacher_a_logits, teacher_b_logits]
weights = [0.7, 0.3]

mixed = mix_teacher_logits(
    teacher_logits_list, weights,
    mode="arithmetic",  # hoặc "geometric"
)
teacher = PrecomputedTeacher(logits=mixed)
```

`mode="arithmetic"` trung bình xác suất; `mode="geometric"` là product-of-experts.

### 6.3 Span-aligned likelihoods

So sánh log-likelihood mỗi model gán cho text spans:

```python
from orda_kernel import (
    AlignedSpanPairs, AlignedLikelihoodConfig,
    aligned_span_distillation_loss, build_precomputed_span_teacher,
)

# Alignment từ external tokenizer offset matching
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
        objective="binary_kl",  # hoặc "binary_ce"
    ),
)
```

Spans không có valid tokens bị drop. Alignment (DP matching, byte-offset mapping) xảy ra bên ngoài package.

**Khi nào dùng cái nào:**

- **Token-map:** nhanh, chỉ exact-match, mất kiến thức teacher không khớp
- **Multi-teacher:** ensemble nhiều cached teachers, không có vocabulary constraint
- **Span-aligned:** xử lý tokenizer differences tùy ý, yêu cầu span alignment preprocessing

---

## 7. Autotune

Lần thực thi đầu benchmarks launch configurations và cache winner in-memory và on disk (SQLite). Cache key: device fingerprint, workload shape, dtype, pipeline identity, và timed-source AST digest. KL hyperparameters và temperatures là runtime scalars — sweep chúng không bao giờ re-benchmark.

**Cache location:** `ORDA_AUTOTUNE_CACHE_DIR`, ngược lại platform cache dir (`~/.cache/orda_kernel` trên Linux, `%LOCALAPPDATA%/orda_kernel` trên Windows).

**Trên clusters:** trỏ `ORDA_AUTOTUNE_CACHE_DIR` đến node-local storage. SQLite WAL + OS locking không đáng tin cậy trên mọi NFS.

**Autotune modes:**

| Mode | Hành vi |
|---|---|
| `autotune=False` | Dùng default config; không benchmark |
| `autotune=True` | Screen candidates, advance finalists đến robust timing |
| `autotune="Max"` | Đo sâu hơn với fixed warmup/rep controls |

`refresh_autotune=True` re-benchmark một lần mỗi process và workload, tôn trọng độ sâu của effective `autotune` mode.

**Tắt autotune:**

```python
config = KernelConfig(autotune=False)
```

**Xóa cache:**

```python
import orda_kernel
orda_kernel.reset_autotune_cache()
```

Xem [reference.md §12](reference.md#12-biến-môi-trường-autotune) cho tất cả environment knobs.

---

## 8. Lỗi phổ biến

**1. Quên rằng tied teacher train theo mặc định**

```python
# teacher_ce_weight mặc định 1.0 ở đây — teacher half nhận CE gradient
out = distillation_loss(
    student_hidden, weight, labels,
    TiedTeacher(teacher_hidden),
    kl_weight=1.0,
)
```

Với tied teachers `teacher_ce_weight=None` resolve về `1.0` (trained path); với separate và precomputed nó resolve về `0.0`. Truyền `teacher_ce_weight=0.0` cho frozen tied teacher, và thêm `mode="frozen"` nếu bạn muốn ý định đó được kiểm tra.

**2. Quên detach cached teacher tensors**

```python
# Sai: raises "cached tensors must not require_grad"
teacher = PrecomputedTeacher(logits=teacher_logits)  # teacher_logits.requires_grad=True
```

Dùng `with torch.no_grad()` hoặc `.detach()` trước khi construct `PrecomputedTeacher`.

**3. Tái sử dụng precomputed teacher sau in-place mutation**

```python
teacher = PrecomputedTeacher(logits=cached_logits)
cached_logits.add_(1.0)  # ← in-place mutation
out = distillation_loss(..., teacher)  # raises ValueError
```

Construct descriptor mới thay vì.

**4. Mong đợi chỉ `GradScaler` là đủ để chống underflow fp16 trên đường fused**

```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    out = distillation_loss(..., kl_weight=1.0)  # ← thiếu grad_scaler=
scaler.scale(out.loss).backward()
```

Loss scale có tới `backward()`, nhưng kernel này đã lưu gradients xuống fp16 từ trong `forward()`; các giá trị underflow lúc đó đã là zero. Hãy truyền `grad_scaler=scaler` để hệ số được áp trước lần ghi — xem [mục 4](#4-mixed-precision-và-gradient-scaling).

**5. Truyền cả `gradient_materialization_scale` và `grad_scaler`**

```python
# Sai: loại trừ lẫn nhau
out = distillation_loss(
    ...,
    gradient_materialization_scale=scale,
    grad_scaler=scaler,  # ← raises ValueError
)
```

Chọn một hoặc không có.
