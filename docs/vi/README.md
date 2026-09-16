# orda-kernel

🌐 **Ngôn ngữ (Languages):** [Tiếng Việt](README.md) | [English](../../README.md)

Triton kernel cho cross-entropy và KL-divergence knowledge distillation. `orda_kernel` hợp nhất student CE, teacher CE và KL term vào một kernel chia khối, không tạo tensor `(BT, vocab)` đầy đủ — chi phí bộ nhớ chi phối của distillation trên từ vựng lớn.


**Năng lực cốt lõi**

- Ba chế độ teacher: tied (shared head), separate (independent head), precomputed (cached)
- 15 biến thể KL divergence: forward, reverse, Jeffreys, mutual, cojeffreys, alpha-divergence, skew-forward/reverse/JS, entropy-adaptive, Rényi, DKD, ABKD, triangular
- Nhiệt độ softmax độc lập cho student và teacher với các biến thể detached

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| README này | Cài đặt, hướng dẫn nhanh, bề mặt API |
| [guide.md](guide.md) | Hướng dẫn sử dụng theo tác vụ: chọn teacher mode và variant, mixed precision, tinh chỉnh bộ nhớ |
| [reference.md](reference.md) | API đầy đủ: hàm, tham số, ràng buộc, lỗi, biến môi trường |
| [variants.md](variants.md) | Biến thể thực thi: đường dẫn kernel, biến thể KL, tương thích, giới hạn |
| [architecture.md](architecture.md) | Luồng thực thi: layout, planning, chunking, autotune |
| [designer.md](designer.md) | Hướng dẫn thay đổi kernel: contracts, extension checklists |
| [gpu-validation.md](gpu-validation.md) | Lệnh validation CUDA/HIP |

---

## Cài đặt

Yêu cầu `torch>=2.3.0` và Python `>=3.10`.

```bash
pip install orda-kernel
```

**GPU (CUDA):** Thêm Triton với:

```bash
pip install 'orda-kernel[gpu]'
```

Triton kernels chạy tự động trên CUDA tensor với dtype được hỗ trợ (`float16`, `float32`, `bfloat16` trên phần cứng BF16 native).

**GPU (ROCm):** Cài ROCm Torch trước (đi kèm Triton riêng), sau đó cài ORDA mà không động dependencies: `pip install --no-deps orda-kernel`. Runtime phát hiện phân phối Triton tự động. Xem [gpu-validation.md](gpu-validation.md) để kiểm chứng.

**CPU:** Không có Triton path. Các lệnh gọi chạy bản Torch, tạo logits đầy đủ — dùng cho phát triển và kiểm thử, không đạt được profile bộ nhớ mà kernel hướng tới.

```python
import orda_kernel
orda_kernel.is_available()  # True khi Triton + CUDA/HIP kernels có thể chạy
```

Optional extras: `test` (pytest), `ci` (ruff, mypy, pytest plugins), `gpu` (triton). Wheels bao gồm marker PEP 561 `py.typed`.

**Cài từ source:**

```bash
git clone https://github.com/hiwuhgds-pixel/ORDA-Knowledge-Distillation-Kernel orda-kernel
cd orda-kernel
pip install .
```

---

## Hướng dẫn nhanh

Đầu vào:

- `student_hidden`: `(BT, hidden)` — hidden states batch × seq đã flatten
- `weight`: `(vocab, hidden)` — student output projection
- `labels`: `(BT,)`, `torch.long` — targets, với `ignore_index` cho vị trí bị mask
- `teacher`: một trong ba teacher descriptors

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

`distillation_loss` trả về `DistillationLossOutput(loss, student_ce, teacher_ce, kl)`. Các components unweighted cho logging. Mỗi fused Triton forward hỗ trợ một backward pass.

### Teacher descriptors

| Descriptor | Fields | Ý nghĩa |
|---|---|---|
| `TiedTeacher(hidden, mode=None)` | `hidden: (BT, hidden)` | Student và teacher dùng chung projection `weight`. `mode` tùy chọn (`"trained"`/`"frozen"`/`"cotrained"`) khẳng định vai trò mà không thay đổi routing. |
| `SeparateTeacher(hidden, weight)` | `hidden: (BT, D_t)`, `weight: (vocab, D_t)` | Teacher có hidden states và projection độc lập. Gradients chảy đến tensor có `requires_grad=True`. |
| `PrecomputedTeacher(logits=… \| teacher_hidden=…, teacher_weight=…)` | `logits: (BT, vocab)` hoặc `teacher_hidden: (BT, D_t)` + `teacher_weight: (vocab, D_t)` | Cached teacher outputs. Không có gradient đến teacher; cached tensors phải không `require_grad`. Phát hiện in-place mutation sau construction. |

Vai trò teacher xuất phát từ weights: `kl_divergence ∈ {"mutual","cojeffreys"}` với `kl_weight > 0` co-train teacher; ngược lại `teacher_ce_weight > 0` train qua CE; ngược lại frozen. `teacher_ce_weight=None` mặc định `1.0` cho tied teachers và `0.0` cho separate/precomputed. Xem [guide.md](guide.md#1-chọn-teacher-mode) để biết cách chọn, và [variants.md §1](variants.md#1-cách-một-call-định-tuyến-biến-thể) cho bảng routing.

---

## Tham số

`distillation_loss(student_hidden, weight, labels, teacher, *, ...)`

Xem [reference.md §1](reference.md#1-distillation_loss) cho ràng buộc đầy đủ.

| Keyword | Mặc định | Mô tả |
|---|---|---|
| `student_ce_weight` / `teacher_ce_weight` / `kl_weight` | `1.0` / `None` / `0.0` | Trọng số term (finite, `>= 0`). |
| `kl_temperature` | `1.0` | Nhiệt độ softmax student; mỗi KL row scale bởi `T_s**2`. |
| `kl_teacher_temperature` | `None` | Nhiệt độ teacher (mặc định `T_s`). Cho phép với teacher-detached variants; `mutual`/`cojeffreys` yêu cầu `T_t == T_s`. |
| `kl_divergence` | `"forward"` | Một trong 15 variants. |
| `kl_alpha`, `kl_beta`, `entropy_gate` | `None` | Variant hyperparameters với phạm vi theo từng variant. |
| `ignore_index` | `-100` | Giá trị label loại trừ khỏi CE/KL; không bao giờ dùng làm memory index. |
| `reduction` | `"mean"` | `"mean"` hoặc `"sum"`. |
| `profile` | `"balanced"` | `"balanced"`, `"fast"`, `"debug"`. |
| `backend` | `"auto"` | `"auto"`, `"triton"`, `"torch"`. |
| `config` | `None` | `KernelConfig` overrides: chunking, autotune mode, weight-gradient accumulation, `assume_valid_labels`. |
| `gradient_materialization_scale` / `grad_scaler` | `None` / `None` | Kiểm soát độ chính xác khi fused path lưu gradient dưới fp16 — xem [guide.md §4](guide.md#4-mixed-precision-và-gradient-scaling). |

## KL variants

`p` = student, `q` = teacher. Tất cả variants trừ `mutual`/`cojeffreys` detach teacher.

Xem [variants.md §4](variants.md#4-các-biến-thể-kl-divergence) cho công thức, phạm vi, zero-support behavior.

| Variant | Objective | Tham số |
|---|---|---|
| `forward` | `KL(q‖p)` | — |
| `reverse` | `KL(p‖q)` | — |
| `jeffreys` | `KL(q‖p) + KL(p‖q)` | — |
| `mutual` | mutual-learning KL (teacher nhận gradient) | — |
| `cojeffreys` | symmetric co-training KL (teacher nhận gradient) | — |
| `alpha_divergence` | power α-divergence | `kl_alpha ∈ (0,1)` |
| `skew_forward` | `KL(q‖(1−β)q+βp)` | `kl_beta ∈ (0,1)` |
| `skew_reverse` | `KL(p‖(1−β)q+βp)` | `kl_beta ∈ (0,1)` |
| `skew_js` | skew Jensen–Shannon | `kl_beta ∈ (0,1)` |
| `entropy_adaptive` | teacher-entropy-gated forward/reverse blend | `entropy_gate` |
| `renyi` / `renyi_reverse` | Rényi `D_r(q‖p)` / `D_r(p‖q)` | `kl_alpha = r ∈ (0,2]`, `\|r−1\| ≥ 0.01` |
| `dkd` | `λ_t·TCKD + λ_n·NCKD` | `kl_alpha=λ_t`, `kl_beta=λ_n ∈ (0,16]` |
| `abkd` | AB-divergence | `kl_alpha`, `kl_beta ∈ [0.05,2]` |
| `triangular` | triangular discrimination | — |

`mutual` và `cojeffreys` yêu cầu `TiedTeacher` hoặc `SeparateTeacher` và route đến cotrained kernel paths. Với `TiedTeacher`, shared projection weight tích lũy gradient từ cả student và teacher halves.

---

## Chưng cất chéo từ vựng

Khi student và teacher dùng tokenizer khác nhau, per-column KL không xác định. Ba công cụ xử lý:

**Token-map projection → `PrecomputedTeacher`**

`build_exact_token_map(student_vocab, teacher_vocab)` + `project_teacher_logits(...)` chiếu teacher distribution lên student tokens khớp. Duplicates sum, unmatched columns nhận `-inf`. Truyền `temperature=T` để temper trước aggregation và khai báo với `PrecomputedTeacher(logits=..., logits_temperature=T)`.

**Multi-teacher mixing → `PrecomputedTeacher`**

`mix_teacher_logits([...], weights, mode="arithmetic"|"geometric")` trộn cached teacher distributions (probability average hoặc product-of-experts). `generalized_jensen_shannon` cung cấp per-row agreement diagnostic.

**Span-aligned likelihoods**

`aligned_span_distillation_loss` so sánh log-likelihood mỗi model gán cho text spans:

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

Spans không có valid tokens bị drop. Alignment (tokenizer offsets, DP matching) xảy ra bên ngoài package.

Xem [guide.md §6](guide.md#6-chưng-cất-chéo-từ-vựng) để chọn giữa ba cách.

---

## Backends

| `backend` | Hành vi |
|---|---|
| `"auto"` (mặc định) | Fused Triton kernel khi có Triton, thiết bị CUDA/HIP, dtype được hỗ trợ và policy được hỗ trợ; ngược lại dùng bản Torch kèm một warning nêu lý do. |
| `"triton"` | Bắt buộc fused path; raise thay vì fall back. |
| `"torch"` | Luôn dùng bản tham chiếu; tạo logits `(BT, vocab)` đầy đủ. |

Hỗ trợ dtype: Triton `{fp16, bf16, fp32}` (bf16 cần phần cứng native); Torch thêm `fp64`.

---

## Testing

```bash
pip install '.[test]'
python test/run_test_cpu.py
```

CPU runner thực thi unit, CPU-reference, và multiprocess tests. GPU correctness và autotune suites thuộc `test/run_test_gpu.py`:

```bash
python test/run_test_gpu.py --test public-api
python test/run_test_gpu.py --route tied --kl forward
```

Xem [gpu-validation.md](gpu-validation.md) cho full GPU gate.

Lint và type-check:

```bash
pip install '.[ci]'
python -m ruff format --check src test scripts .github/scripts benchmark
python -m ruff check src test scripts .github/scripts benchmark
python -m mypy
```
