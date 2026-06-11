# orda_ce_kernel

Triton kernel cho ORDA KL distillation + fused Cross Entropy substrate.

Ngôn ngữ tài liệu: [English](README.md) | [Tiếng Việt](README_VI.md)

Người dùng mô tả cấu hình distillation bằng các teacher object và loss weight.
Các tùy chọn kernel được gom trong `KernelConfig` hoặc chọn qua preset `profile`.

## Trạng thái hiện tại

Repository này chưa phải là một thư viện huấn luyện có thể cắm vào dùng ngay
trong hầu hết trường hợp. Phiên bản hiện tại mới triển khai phần khung lõi của
ORDA KL distillation kernel và fused Cross Entropy substrate đi kèm, nhưng phạm
vi kiểm chứng vẫn bị giới hạn bởi tài nguyên compute hiện có của tác giả.

Toàn bộ test và benchmark artifact hiện tại được tạo trên tài nguyên miễn phí
Tesla T4. Do không có quyền truy cập rộng hơn tới nhiều GPU, dự án vẫn cần thêm
kiểm chứng về độ đúng, độ ổn định và hiệu năng trên các kiến trúc GPU khác nhau
trước khi có thể xem như một dependency drop-in đã trưởng thành.

Repository này được public với hy vọng cộng đồng có thể hỗ trợ chạy test,
benchmark, báo lỗi và thử nghiệm trên nhiều thiết bị CUDA/HIP hơn. Những phản
hồi đó sẽ giúp cải thiện tính portable của kernel và hoàn thiện thêm một hướng
kernel mới cho knowledge distillation, một nhóm kernel open implementation hiện
vẫn còn tương đối khan hiếm.

## Cài đặt

```bash
python -m pip install -e .
```

Để chạy validation cục bộ và các kiểm tra phát triển:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/unit -q
python scripts/check_ast_imports.py
```

Triton execution hiện nhắm tới các thiết bị PyTorch kiểu CUDA và đã được kiểm
chứng trên CUDA/Tesla T4 fp16. Hỗ trợ HIP/ROCm được định hướng thông qua lớp
trừu tượng CUDA/HIP device của PyTorch, nhưng nên xem là experimental cho tới
khi được kiểm chứng trên phần cứng ROCm thật. Các lệnh CPU có thể dùng PyTorch
reference fallback với `backend="auto"` hoặc `backend="torch"`.

## Quickstart

```python
import torch
from orda_ce_kernel import distillation_loss, TiedTeacher

student_hidden = torch.randn(
    2048, 4096, device="cuda", dtype=torch.float16, requires_grad=True
)
teacher_hidden = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
weight = torch.randn(
    32000, 4096, device="cuda", dtype=torch.float16, requires_grad=True
)
labels = torch.randint(0, 32000, (2048,), device="cuda")

out = distillation_loss(
    student_hidden,
    weight,
    labels,
    TiedTeacher(teacher_hidden),
    student_ce_weight=1.0,
    kd_weight=0.4,
    temperature=1.5,
    profile="balanced",
    backend="auto",
)

out.loss.backward()
```

## Validation và benchmark

Hướng dẫn test và benchmark đầy đủ nằm ở
[`tests/TESTING_EN.md`](tests/TESTING_EN.md). Các correctness test CUDA/Triton
cần môi trường GPU thật:

```bash
python -m pytest tests/correctness -q
python scripts/run_all_test_colab.py
```

Các ví dụ benchmark smoke được ghi trong
[`tests/TESTING_EN.md`](tests/TESTING_EN.md#benchmarks). Các con số benchmark
hiện tại nên được chạy lại trên GPU mục tiêu trước khi dùng để đưa ra nhận định
riêng cho phần cứng đó.

## Teacher Modes

```python
from orda_ce_kernel import TiedTeacher, SeparateTeacher, PrecomputedTeacher

teacher = TiedTeacher(hidden=teacher_hidden)
teacher = SeparateTeacher(hidden=teacher_hidden, weight=teacher_weight)
teacher = PrecomputedTeacher(logits=teacher_logits)
```

KL Triton kernel luôn nhận cùng một layout buffer nội bộ:

```text
logits_chunk[0:n_rows]        = student logits
logits_chunk[n_rows:2*n_rows] = teacher logits
```

Teacher object chỉ quyết định buffer đó được tạo như thế nào và gradient path
nào tồn tại. KL phải đọc logits sạch trước khi CE kernel ghi đè cùng buffer đó
in-place thành CE gradients.

## Chiến lược giảm VRAM

Phần tiết kiệm bộ nhớ chính đến từ việc tránh tạo các tensor phân phối KL trên
toàn bộ batch-token. Một đường tính KL PyTorch thông thường sẽ materialize
student log-probabilities và teacher probabilities trên toàn bộ chiều
batch-token:

```text
log_p_s: [BT, V]
p_t:     [BT, V]
```

Tùy compiled graph, có thể còn xuất hiện thêm một tensor KL elementwise với
shape `[BT, V]`.

ORDA dùng fused Cross Entropy path như một substrate cho KL. Ở mỗi chunk, kernel
trước tiên tạo một logits buffer dùng chung:

```text
logits_chunk[0:n_rows]        = student logits
logits_chunk[n_rows:2*n_rows] = teacher logits
```

KL Triton kernel đọc trực tiếp buffer logits sạch này trước khi CE kernel ghi đè
nó bằng CE gradients. Kernel tính KL theo từng row và student KL gradient, sau
đó chỉ materialize buffer gradient riêng của KL:

```text
grad_kl_student: [n_rows, V]
```

Gradient này được cộng ngược vào `logits_chunk[:n_rows]` trước khi buffer dùng
chung tiếp tục đi qua lifecycle backward của CE/GEMM.

Ví dụ, trong case benchmark T4
[`dim=1024 vocab=128k seq=512`](simulate/bench_vram_TiedTeacher.txt), với
`batch=16`, `BT=8192`, và `V=131072`, các tensor KL cốt lõi của PyTorch là:

```text
log_p_s [8192, 131072] fp16 = 2048 MiB
p_t     [8192, 131072] fp16 = 2048 MiB
total core KL tensors        = 4096 MiB
```

Với ORDA dynamic chunking, `num_chunks=16` và `n_rows=512`, nên buffer extra
riêng của KL là:

```text
grad_kl_student [512, 131072] fp16 = 128 MiB
```

Shared CE+KL logits buffer cho chunk đó là:

```text
logits_chunk [1024, 131072] fp16 = 256 MiB
```

Benchmark đo được:

```text
torch-compile CE+KL = 8480.3 MB
orda CE+KL          = 1223.6 MB
```

Các giá trị đo được ở trên là peak extra memory cho toàn bộ CE+KL loss backend,
bao gồm CE logits, backward buffers, GEMM workspace, gradients, và allocator
overhead. Thay đổi từ `2 * [BT, V]` xuống `[n_rows, V]` ở trên là phần giảm bộ
nhớ KL cốt lõi, chưa tính bất kỳ tensor KL elementwise `[BT, V]` bổ sung nào mà
compiled graph có thể materialize.

## Scope và Teacher Modes

Thư viện này là một logit-level forward-KL distillation kernel với fused Cross
Entropy substrate. Nó tập trung quanh response-based KD ở output layer và cung
cấp ba teacher mode để tạo hoặc cung cấp teacher logits. Feature-based KD,
relation-based KD, attention transfer, reverse KL, JS divergence, MSE-on-logits,
và sequence-level KD nằm ngoài scope thiết kế hiện tại.

Cả ba teacher mode đều tạo cùng layout input nội bộ cho KL:

```text
logits_chunk[0:n_rows]         = student logits
logits_chunk[n_rows:2*n_rows]  = teacher logits
```

Các mode khác nhau ở cách teacher logits được tạo và gradient path nào có sẵn.

### `TiedTeacher(hidden)`

`TiedTeacher` là shared-head mode. Student và teacher hidden states được project
bằng cùng một output weight:

```text
student logits = student_hidden @ weight.T
teacher logits = teacher_hidden @ weight.T
```

Trong Triton path, student và teacher hidden chunks được concatenate rồi project
bằng một shared GEMM. Mode này tương thích với các pattern self-distillation
hoặc online KD nơi teacher signal được kỳ vọng dùng chung vocabulary head với
student.

Mặc định, `teacher_ce_weight=None` được resolve thành `1.0` cho mode này, nên
teacher CE branch được bật trừ khi tắt rõ ràng. Shared `weight` có thể nhận
gradient từ student CE branch, teacher CE branch, và student side của KL. Bản
thân KL là teacher-detached: KL gradient chỉ được cộng vào student logits slice.

### `SeparateTeacher(hidden, weight)`

`SeparateTeacher` là separate-head mode. Student và teacher logits được tạo bằng
hai projection weight khác nhau:

```text
student logits = student_hidden @ weight.T
teacher logits = teacher_hidden @ teacher_weight.T
```

Mode này map trực tiếp tới các setup nơi teacher có output head riêng, chẳng hạn
external pretrained teacher, EMA teacher, teacher có hidden dimension khác, hoặc
runtime teacher branch.

Mặc định, `teacher_ce_weight=None` được resolve thành `0.0`, nên mode này chạy
như student CE + forward KL trừ khi teacher CE được bật rõ ràng. Student weight
có thể nhận gradient từ student CE và KL. Teacher hidden state và teacher weight
chỉ có thể nhận gradient từ teacher CE branch khi `teacher_ce_weight > 0`; chúng
không nhận KL gradient vì teacher distribution được detach.

### `PrecomputedTeacher(logits)`

`PrecomputedTeacher` là cached-logits mode. Teacher logits được cung cấp trực
tiếp dưới dạng tensor `[BT, V]`:

```text
teacher logits = supplied_teacher_logits
```

Loss path project student hidden states, rồi concatenate student logits với
teacher logits đã cung cấp để xử lý CE/KL. Teacher logits được cung cấp không
được require gradients. Mode này tương thích với các pipeline offline KD nơi
teacher logits được precompute hoặc được tạo bên ngoài loss call hiện tại.

Mặc định, `teacher_ce_weight=None` được resolve thành `0.0`. Nếu dùng
`teacher_ce_weight > 0`, teacher CE có thể được report hoặc đưa vào loss từ
supplied logits, nhưng vẫn không có teacher gradient path.

## Các setup Forward-KL có thể biểu diễn bằng các mode này

Các mục dưới đây nên được hiểu là những usage pattern response-based forward-KL
mà các teacher mode hiện tại có thể biểu diễn, không phải các họ KD riêng biệt
được thư viện triển khai.

| Forward-KL setup | Mode tương thích | Ghi chú |
| :--- | :--- | :--- |
| Response-based KD | `TiedTeacher`, `SeparateTeacher`, `PrecomputedTeacher` | Tất cả mode đều hoạt động trên output logits và tính forward KL từ teacher distribution sang student distribution. |
| Offline KD với frozen pretrained teacher | `SeparateTeacher`, `PrecomputedTeacher`; `TiedTeacher` cho shared-head designs | Dùng `SeparateTeacher` khi frozen teacher được evaluate trong training, hoặc `PrecomputedTeacher` khi logits đã được cache. |
| Online KD / self-distillation | `TiedTeacher`; `SeparateTeacher` khi online teacher có head riêng | `TiedTeacher` tương thích với shared-head teacher signals. `SeparateTeacher` có thể biểu diễn online teacher branch có head riêng. |
| Co-distillation / mutual-learning-style training | `SeparateTeacher`; `TiedTeacher` cho shared-head variants | `teacher_ce_weight > 0` bật teacher CE gradients. KL term vẫn teacher-detached, nên kernel không triển khai symmetric mutual KL. |
| EMA teacher / frozen separate head | `SeparateTeacher`; `PrecomputedTeacher` cho cached EMA logits | `SeparateTeacher` có thể dùng EMA/frozen teacher head được evaluate tại runtime. `PrecomputedTeacher` có thể dùng logits được tạo ngoài loss call. |

## Tài liệu tham khảo và công trình liên quan

Dự án này dựa trên một số ý tưởng đã có trong knowledge distillation và các
loss kernel hiệu quả:

- Logit-level knowledge distillation đi theo setup response-based distillation
  được giới thiệu bởi Hinton et al. trong
  [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
- Đường online softmax tùy chọn đi theo ý tưởng online normalizer của Milakov và
  Gimelshein,
  [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) là công trình có mức
  tương quan trong ngữ cảnh các Triton loss kernel tiết kiệm bộ nhớ, đặc biệt
  với Cross Entropy kernels. Phần giao nhau nằm ở các hướng kỹ thuật chung như
  chia chunk và lưu gradient in-place. ORDA áp dụng các hướng này trong một
  lifecycle teacher-student forward-KL distillation riêng.

## Tùy chỉnh nâng cao

```python
from orda_ce_kernel import KernelConfig

config = KernelConfig(
    online_softmax=True,
    fast_math=False,
    quantize_grad_weight=False,
    stochastic_rounding=False,
    fp32_grad_weight_accumulation=False,
    max_chunks=None,
)
```

Truyền `config=` sẽ override `profile=`.

`profile="fast"` chỉ bật fast math. INT8 gradient compression và stochastic
rounding là opt-in. Đặt `stochastic_seed` để stochastic quantization có thể tái
lập.

`max_chunks`: Tham số giới hạn số lượng chunk tối đa được chia. Khi cấu hình là `None`, hệ thống tự động tính toán giới hạn và chỉ cho phép nhân đôi số lượng chunk (lũy thừa của 2) tối đa 1 lần để giảm kích thước chunk khi xảy ra lỗi tràn bộ nhớ (OOM). Hiện tại, thuật toán heuristic của kernel đang chia chunk ở mức 512-1024 mỗi chunk dựa theo kết quả nhanh nhất trong các bài test trên Tesla T4. Nhưng nó có thể không đúng đối với mọi loại GPU, đặt biệt là khi sử dụng với kích thước BT rất lớn khiến chunk được chia rất nhiều mà hiệu quả mang lại không còn cao.  Khuyến nghị người dùng nên chủ động cấu hình giá trị `max_chunks` phù hợp dựa trên dung lượng VRAM khả dụng.

`student_ce`, `teacher_ce`, và `kl` là các reported component đã detach. Chỉ
`loss` mang backward path. `reduction="mean"` chia CE và KL theo số token hợp
lệ; `reduction="sum"` trả về các component CE và KL dạng tổng.

CUDA execution cast kernel compute path sang fp16. HIP/ROCm path được định
hướng cast kernel compute buffers sang bf16, nhưng validation và tài liệu hiệu
năng hiện tại vẫn scoped cho T4/fp16 trừ khi một benchmark GPU cụ thể nói khác.
Hỗ trợ input dtype không có nghĩa là kernel compute path chạy full fp32.

Với `PrecomputedTeacher`, teacher logits được xem là hằng số đã cung cấp. Khi
`teacher_ce_weight > 0`, `teacher_ce` được report nhưng không có gradient path.

`profile="debug"` chọn cấu hình numerical-reference/debug.
