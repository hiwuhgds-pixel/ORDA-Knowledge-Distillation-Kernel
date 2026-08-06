# Tài liệu tham khảo API

Contract public đầy đủ: mọi export, tham số, ràng buộc, và lỗi. Về cách dùng theo tác vụ, xem [guide.md](guide.md). Về routing tới các kernel path và mức hỗ trợ của từng biến thể, xem [variants.md](variants.md).

Giao diện public (`import orda_kernel`):

| Nhóm | Exports |
|---|---|
| API Tokenwise | `distillation_loss`, `DistillationLoss`, `DistillationLossOutput` |
| Teacher descriptors | `TiedTeacher`, `SeparateTeacher`, `PrecomputedTeacher` |
| Cấu hình | `KernelConfig`, `ChunkSize`, `EntropyGate` |
| API Aligned-span | `aligned_span_distillation_loss`, `AlignedSpanPairs`, `AlignedLikelihoodConfig`, `CrossVocabTeacher`, `PrecomputedSpanTeacher`, `build_precomputed_span_teacher` |
| Builders precomputed-input | `build_exact_token_map`, `project_teacher_logits`, `mix_teacher_logits`, `generalized_jensen_shannon` |
| Điều khiển runtime | `is_available`, `reset_autotune_cache`, `begin_autotune_run`, `__version__` |

Các low-level op (`orda_kernel.ops.*`) được ghi tại [mục 8](#8-low-level-ops).

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

### Input vị trí

| Tham số | Kiểu / shape | Contract |
|---|---|---|
| `student_hidden` | tensor float `(BT, H)` | 2D; hidden states `batch * seq` đã flatten. Xác định device tham chiếu. |
| `weight` | tensor float `(V, H)` | Output projection của student; `V >= 1`; cùng device và (ngoài autocast) cùng dtype với `student_hidden`. |
| `labels` | `(BT,)` `torch.long` | Giá trị trong `[0, V)` hoặc chính xác `ignore_index`; cùng device. Kiểm tra phạm vi chặt theo mặc định (báo tối đa 5 giá trị vi phạm). |
| `teacher` | `TiedTeacher \| SeparateTeacher \| PrecomputedTeacher` | Xem [mục 4](#4-teacher-descriptors). |

### Tham số keyword

| Keyword | Kiểu | Mặc định | Contract |
|---|---|---|---|
| `student_ce_weight` | `float` | `1.0` | Hữu hạn, `>= 0`. `bool` bị từ chối. Trọng số của term student CE. |
| `teacher_ce_weight` | `float \| None` | `None` | Hữu hạn, `>= 0` khi được cung cấp. `None` giải quyết theo mode: tied → `1.0`, separate/precomputed → `0.0`. `> 0` chọn một trained path. |
| `kl_weight` | `float` | `0.0` | Hữu hạn, `>= 0`. `!= 0` kích hoạt term KL. |
| `kl_temperature` | `float` | `1.0` | Hữu hạn, `> 0`; `1/T` và `T**2` phải là giá trị float32 normal. |
| `kl_teacher_temperature` | `float \| None` | `None` | `None` → bằng `kl_temperature`. Cùng ràng buộc số học (`1/T_t` float32 normal). Phải bằng `kl_temperature` với `mutual`/`cojeffreys`. |
| `kl_divergence` | `KLVariant` | `"forward"` | Một trong 15 biến thể ([variants.md §4](variants.md#4-các-biến-thể-kl-divergence)). `mutual`/`cojeffreys` yêu cầu teacher tied hoặc separate (bị từ chối với precomputed kể cả khi không hoạt động). |
| `kl_alpha` | `float \| None` | `None` | Phạm vi phụ thuộc biến thể: `alpha_divergence` `(0,1)`; `renyi`/`renyi_reverse` `(0,2]` loại trừ `\|r−1\| < 0.01`; `dkd` `(0,16]`; `abkd` `[0.05,2]`. Khi biến thể không đọc nó, giá trị cung cấp vẫn phải hữu hạn trong `(0,1)`. Khi KL không hoạt động, phạm vi riêng của biến thể đã chọn được dùng để kiểm tra trước khi canonicalize thành `0.5`. |
| `kl_beta` | `float \| None` | `None` | `skew_forward`/`skew_js`/`skew_reverse` `(0,1)`; `dkd` `(0,16]`; `abkd` `[0.05,2]`. Cùng quy tắc canonicalize như `kl_alpha`; KL không hoạt động vẫn kiểm tra theo phạm vi `kl_beta` của biến thể đã chọn. |
| `entropy_gate` | `EntropyGate \| None` | `None` | Chỉ `entropy_adaptive` đọc (mặc định `EntropyGate()` khi không đặt); kiểm tra kiểu cho mọi biến thể. |
| `ignore_index` | `int` | `-100` | Bất kỳ số nguyên (`bool` bị từ chối). Không bao giờ dùng làm memory index. |
| `reduction` | `"mean" \| "sum"` | `"mean"` | `mean` chia cho số label không bị ignore, chặn dưới `>= 1`. |
| `profile` | `"balanced" \| "fast" \| "debug"` | `"balanced"` | Cơ sở `KernelConfig` có sẵn — xem [mục 6](#6-profiles). |
| `backend` | `"auto" \| "triton" \| "torch"` | `"auto"` | Xem [mục 7](#7-backends). |
| `config` | `KernelConfig \| None` | `None` | Các override theo trường, hợp nhất lên trên profile — xem [mục 5](#5-kernelconfig). |
| `gradient_materialization_scale` | scalar tensor \| `None` | `None` | Tensor `float32` một phần tử trên device student, không có gradients. Được nhân vào logit gradients ngay trong kernel lúc tạo chúng và chia ra lại ở backward, nên gradients không đổi. Chống underflow fp16 mà một `GradScaler` ở mức loss không thể với tới — xem [Gradient scaling dưới fp16](#gradient-scaling-dưới-fp16). Bị bỏ qua bởi `backend="torch"`. Loại trừ lẫn nhau với `grad_scaler`. |
| `grad_scaler` | `torch.amp.GradScaler \| None` | `None` | Khi bật, dẫn xuất `gradient_materialization_scale` từ scaler; một scaler đã tắt là no-op. Nó không scale loss trả về — vẫn gọi `scaler.scale(loss).backward()` như thường. Xem [Gradient scaling dưới fp16](#gradient-scaling-dưới-fp16). |

**Thứ tự ưu tiên:** với mọi trường có trên cả hai, giá trị `KernelConfig` khác `None` ghi đè tham số call (`kl_weight`, `kl_temperature`, `kl_teacher_temperature`, `kl_divergence`, `kl_alpha`, `kl_beta`, `entropy_gate`), và giá trị `KernelConfig` khác `None` ghi đè profile (`grad_weight_accumulation`, `max_fused_size`, `autotune`, `refresh_autotune`, `chunk_size`). `num_chunks` và `max_chunks` chỉ tồn tại trên `KernelConfig`.

### Giá trị trả về

`DistillationLossOutput`, một `NamedTuple`:

```python
class DistillationLossOutput(NamedTuple):
    loss: torch.Tensor        # tổng có trọng số — backprop qua đây
    student_ce: torch.Tensor  # chưa nhân trọng số, detached
    teacher_ce: torch.Tensor  # chưa nhân trọng số, detached
    kl: torch.Tensor          # chưa nhân trọng số, detached
```

`loss = student_ce_weight·student_ce + teacher_ce_weight·teacher_ce + kl_weight·kl`. Thành phần có trọng số `0` không bao giờ được tính và đọc chính xác `0` (không phải giá trị lẽ ra của nó). Output dtype là `promote_types(input_dtype, float32)`.

**Autograd:** mỗi fused Triton forward hỗ trợ một backward pass (gradients đã cache bị tiêu thụ tại chỗ); hãy tính lại loss thay vì dùng `retain_graph=True`. Torch backend không có giới hạn một backward, nhưng `alpha_divergence`, `renyi`, `renyi_reverse`, và `abkd` dùng các proxy gradient bậc một đã ổn định hóa. Yêu cầu gradient bậc cao hơn cho bốn biến thể đó sẽ báo lỗi thay vì âm thầm trả về một Hessian bỏ qua độ cong KL. Mọi entry point đều `torch._dynamo.disable`d.

### Gradient scaling dưới fp16

Chỉ dùng `torch.amp.GradScaler` là không đủ để chống underflow trên đường fused. Lý do nằm ở *thời điểm* mỗi scale được áp dụng.

Một layer fp16 thông thường sinh gradients trong `backward()`. `GradScaler` thổi phồng loss từ trước, nên độ phồng đó đã nằm sẵn trong chuỗi đạo hàm vào lúc bất kỳ giá trị gradient nào được ghi — các giá trị nhỏ tới fp16 khi đã được nhân sẵn và vẫn biểu diễn được.

Fused kernel không theo thứ tự đó. Nó tính logit gradients **ngay trong `forward()`**, trong cùng lượt quét sinh ra loss, và lưu chúng vào buffer dtype tính toán (fp16 khi input là fp16) để tránh tạo thêm một tensor `(BT, vocab)` thứ hai. `backward()` của nó chỉ rescale lại buffer đã lưu đó. Do vậy:

1. `forward()` ghi gradients xuống fp16; các giá trị dưới ngưỡng subnormal bị đẩy về đúng zero.
2. Hệ số của scaler đến sau, trong `backward()`.
3. Nhân một zero đã lưu với bất kỳ hệ số nào vẫn là zero — tín hiệu đã mất hẳn, không phải chỉ bị thu nhỏ.

`gradient_materialization_scale` lấp khoảng trống đó bằng cách áp một hệ số đúng vào thời điểm duy nhất có tác dụng: bên trong kernel, trên giá trị còn nằm ở thanh ghi float32, *trước* lần ghi xuống fp16. Backward chia đúng hệ số đó ra lại, nên gradient tới tham số về mặt số học chính là kết quả fp32.

Hai cơ chế bổ sung cho nhau chứ không thừa. `GradScaler` bảo vệ mọi layer *khác* trong model; còn `grad_scaler=` là đường tắt tiện lợi đọc hệ số hiện tại của scaler và chuyển vào đây làm materialization scale. Vì cả hai tham số cùng lấp một chỗ, truyền đồng thời sẽ raise `ValueError`.

Hệ số hiệu dụng còn gộp cả rút gọn token-mean, và phải hữu hạn, normal trong dtype lưu trữ gradient. Khi không thỏa, kernel chọn một hệ số trung tính về số học ngay trên device và dời phần rút gọn về backward — không quét host, không đồng bộ. Gradients vẫn đúng, nhưng lớp chống underflow không hoạt động cho lời gọi đó. `backend="torch"` lấy đạo hàm qua autograd mà không tạo buffer gradient vật chất hóa, nên không scale nào được áp và gradients bit-identical dù có hay không.

Xem [guide.md §4](guide.md#4-mixed-precision-và-gradient-scaling) cho các công thức dùng trong vòng lặp huấn luyện.

---

## 2. `DistillationLoss` (nn.Module)

Lưu mọi keyword của `distillation_loss` lúc construction; `forward(student_hidden, weight, labels, teacher, gradient_materialization_scale=None, grad_scaler=None)` chuyển tiếp cấu hình đã lưu cộng gradient scale theo bước. Không tham số nào là tensor học được; module là một container tiện lợi, và các attribute có thể gán lại giữa các call.

---

## 3. Đối tượng hyperparameter

### `EntropyGate`

`EntropyGate(alpha_min=0.0, alpha_max=1.0, tau=0.5, scale=0.1)` — dataclass đóng băng cho gate `entropy_adaptive` `α(h) = alpha_min + (alpha_max − alpha_min) · sigmoid((h − tau) / scale)` với `h` là entropy teacher chuẩn hóa bởi `log V` (`0` khi `V == 1`).

Ràng buộc: mọi trường là float hữu hạn (`bool` bị từ chối), `0 <= alpha_min <= alpha_max <= 1`; cả `scale` và `1 / scale` phải là giá trị float32 normal hữu hạn. Gate tính từ phân phối teacher đã detach và không bao giờ nhận gradient.

### `ChunkSize`

Value object bất biến với ba constructor:

- `ChunkSize.nrows(n)` — chính xác `n` row mỗi chunk (`n` là int dương, clamp về `BT` lúc thực thi).
- `ChunkSize.fast()` — nhắm 1024 row mỗi chunk, chặn ở tối thiểu 2 và tối đa 16 chunk.
- `ChunkSize.dynamic()` — phân hoạch ban đầu chọn từ shape workload.

Int thô, string, `None`, và các sentinel không dương cũ bị từ chối ở mọi nơi cần `ChunkSize`. `is_dynamic`, `is_fast`, và `value` là accessor chỉ đọc (`value` báo lỗi trên một instance dynamic hoặc fast).

---

## 4. Teacher descriptors

### `TiedTeacher(hidden, mode=None)`

- `hidden: (BT, H)` — phải khớp shape student chính xác; `weight` của student được chia sẻ.
- `mode: None | "trained" | "frozen" | "cotrained"` — một kiểm tra nhất quán mang tính khai báo, không phải công tắc routing. Vai trò hiệu lực suy ra từ `teacher_ce_weight`, `kl_weight`, và `kl_divergence`; một mâu thuẫn báo `ValueError`:
  - `"frozen"` yêu cầu `teacher_ce_weight == 0` và không có `mutual`/`cojeffreys` hoạt động;
  - `"trained"` yêu cầu `teacher_ce_weight > 0` và không có `mutual`/`cojeffreys` hoạt động;
  - `"cotrained"` yêu cầu `kl_weight > 0` với `mutual`/`cojeffreys`.

### `SeparateTeacher(hidden, weight)`

- `hidden: (BT, D_t)`, `weight: (V, D_t)` — projection riêng; `D_t` có thể khác `H`, `V` phải khớp vocabulary của student.
- Gradients chảy tới các tensor có `requires_grad=True`, theo chính sách gradient của path (CE cho trained, KL(+CE) cho cotrained).

### `PrecomputedTeacher(logits=None, teacher_hidden=None, teacher_weight=None, trust_cache_immutability=False, logits_temperature=None)`

Cache teacher đã kiểm tra, không gradient. Ràng buộc thực thi lúc construction:

- Chính xác một biểu diễn: `logits: (BT, V)` hoặc `teacher_hidden: (BT, D_t)` + `teacher_weight: (V, D_t)` (cả hai cùng nhau).
- Không cached tensor nào được `require_grad` (dùng `SeparateTeacher` cho teacher live).
- Giá trị `logits`: hữu hạn hoặc `-inf` (zero support); `NaN`/`+inf` bị từ chối. Giá trị `teacher_hidden`/`teacher_weight`: hữu hạn nghiêm ngặt.
- Các logit cache được row-center trong dtype nguồn trước khi chuyển đổi theo chunk; giá trị hữu hạn dưới phạm vi compute dtype được clamp trong khi các sentinel `-inf` giữ chính xác.
- Giá trị non-finite sinh ra từ GEMM hidden/weight đã cache là không hợp lệ và vẫn hiện ra dưới dạng `NaN` loss/gradients hoạt động thay vì một dead cache row.
- Cached tensors không được clone. Một validation stamp (identity, layout, version counter) được chụp lúc construction và kiểm tra lại ở mỗi call; in-place mutation thông thường sau construction báo `ValueError`. `.data` mutation không an toàn nằm ngoài bảo đảm.
- Construction thực hiện các phép rút gọn kiểm tra toàn nội dung; hãy tạo descriptor một lần và giữ nó ngoài bước huấn luyện.
- `trust_cache_immutability=True` cần thiết bất cứ khi nào `_version` không khả dụng, bao gồm inference tensors tạo dưới `torch.inference_mode()` và tensor subclasses không có version counter đọc được.
- `logits_temperature` (mặc định `None`) khai báo logits đã được temper tại temperature đó (hữu hạn `> 0`, chỉ biểu diễn `logits`), yêu cầu `teacher_ce_weight=0.0`, và làm loss không áp thêm temperature teacher. Khi `kl_teacher_temperature` để mặc định, temperature của cache được áp dụng; một `kl_teacher_temperature` truyền tường minh khác với nó bị từ chối. `None` giữ raw logits, nên loss áp temperature teacher của riêng nó.

Kiểm tra shape với student (`(BT, V)` / `(BT, D_t)`+`(V, D_t)`) diễn ra lúc call. Dtype của `logits` có thể khác model dtype (chuyển đổi theo chunk); nó phải là float.

`teacher.prepare_logits(dtype)` thay thế việc chuyển đổi theo chunk lặp lại đó bằng một cache bền vững tường minh:

- `dtype` phải là một trong các Torch model dtype (`float16`, `bfloat16`, `float32`, hoặc `float64`). Dùng dtype đã chọn cho model student.
- Một logits cache đã contiguous với dtype đó trả về cùng descriptor và không cấp phát gì. Ngược lại phương thức row-center trước một phép cast thu hẹp, tạo một tensor contiguous, và trả về một `PrecomputedTeacher` mới.
- Descriptor mới giữ `logits_temperature` cùng các kiểm tra validation stamp/mutation thông thường. Descriptor nguồn vẫn độc lập.
- Preparation không bao giờ ngầm định: hãy giữ descriptor trả về để tái sử dụng và tính toán phần cấp phát `(BT, V)` đầy đủ của nó. Các cache hidden/weight bị từ chối vì phương thức này không tạo một teacher projection.

---

## 5. `KernelConfig`

Dataclass đóng băng chứa các override theo call. Mọi trường mặc định là `None`/"chưa đặt" và chỉ ghi đè khi được đặt:

| Trường | Kiểu | Tác dụng |
|---|---|---|
| `grad_weight_accumulation` | `"auto" \| "compute" \| "fp32" \| None` | Dtype của bộ tích lũy weight-gradient theo chunk. `"auto"` (mặc định) dùng float32 khi dtype tham số khác compute dtype hoặc tham số là float16, và dùng compute dtype trong các trường hợp còn lại; `"compute"` luôn tích lũy trong compute dtype; `"fp32"` luôn tích lũy trong float32. |
| `chunk_size` | `ChunkSize \| None` | `ChunkSize.nrows(n)` (cố định `n` row), `ChunkSize.fast()` (1024 row mỗi chunk, tối thiểu 2 và tối đa 16 chunk), hoặc `ChunkSize.dynamic()` (heuristic theo shape workload). Không đặt thì kế thừa giá trị profile, mặc định về `dynamic()`. Chỉ loại trừ lẫn nhau với `num_chunks` trong trường hợp `nrows`. |
| `num_chunks` | `int \| None` | Yêu cầu tường minh dương. Ghi đè `fast()`, `dynamic()`, và `max_chunks`. |
| `max_chunks` | `int \| None` | Trần ưu tiên thấp hơn cho phân hoạch tự động `fast()` và `dynamic()`; bị bỏ qua bởi `num_chunks` tường minh và `ChunkSize.nrows(n)`. Mặc định là `16` cho `fast()` và `min(32, 2·max(1, BT//512))` cho `dynamic()`, con số này cũng chặn vòng lặp OOM retry của nó. |
| `max_fused_size` | `int \| None` | Lũy thừa của hai `<= 32768`; trần fused block cho mỗi launch. |
| `kl_weight`, `kl_temperature`, `kl_teacher_temperature`, `kl_divergence`, `kl_alpha`, `kl_beta`, `entropy_gate` | như trong call | Ghi đè các tham số call khi được đặt (cùng validation). |
| `autotune` | `bool \| "Max" \| None` | `False` = chỉ dùng mặc định; `True` và `"Max"` dùng cùng tập candidate cho phép bởi `max_fused_size`, trong khi `"Max"` đo các finalist sâu hơn. Cả hai mode đều tôn trọng winner đã cache. `"Max"` dùng các control screen, finalist, và warmup/minimum/maximum của robust stage cố định; các override môi trường tương ứng bị bỏ qua. |
| `refresh_autotune` | `bool \| None` | Benchmark lại một lần cho mỗi workload và process run tại độ sâu đo của mode đã chọn. Không hợp lệ khi `autotune=False` hiệu lực. |
| `assume_valid_labels` | `bool \| None` | `True` bỏ qua bước quét label đồng bộ trên host; indexing an toàn với `ignore_index` phía kernel vẫn hoạt động. |

Lỗi kiểu được báo lúc construction (`chunk_size` phải là `ChunkSize` hoặc `None`, các cờ phải là `bool` hoặc `None`, `grad_weight_accumulation` là một trong các literal của nó hoặc `None`, `entropy_gate` là `EntropyGate` hoặc `None`).

## 6. Profiles

| Profile | `max_fused_size` | `autotune` | `grad_weight_accumulation` | `chunk_size` |
|---|---|---|---|---|
| `balanced` | `16384` | `True` | `"auto"` | chưa đặt → `dynamic()` |
| `fast` | `32768` | `"Max"` | `"auto"` | `fast()` |
| `debug` | `32768` | `False` | `"fp32"` | chưa đặt → `dynamic()` |

Một trường `KernelConfig` luôn thắng giá trị profile của nó. `num_chunks` và `max_chunks` chỉ tồn tại trên `KernelConfig`.

`ChunkSize.fast()` nhắm thông lượng: phân hoạch bỏ qua kích thước vocabulary, giữ ở mức 16 chunk hoặc thấp hơn (các chunk nở rộng vượt `16·1024` row thay vì nhân lên), và không bao giờ retry. Một lỗi out-of-memory báo lỗi kèm phân hoạch nó đã dùng và một `num_chunks` đề xuất, để caller đặt và chọn một phân hoạch mới.

## 7. Backends

| `backend` | Hành vi |
|---|---|
| `"auto"` | Triton khi Triton import được, tensors là CUDA/HIP, dtype được hỗ trợ, và policy có một Triton specialization; ngược lại dùng Torch reference cộng một `RuntimeWarning` một lần trên toàn tiến trình nêu rõ lý do. |
| `"triton"` | Fused path bắt buộc. Báo `RuntimeError` (không có Triton/CUDA), `ValueError` (dtype không hỗ trợ; precomputed teacher CE), hoặc `UnsupportedTritonPolicyError` (policy hợp lệ nhưng không có Triton specialization). |
| `"torch"` | Implementation reference — nguồn chuẩn số học, CPU hoặc GPU, tạo toàn bộ logits `(BT, V)`. Autograd theo chuẩn trừ bốn biến thể proxy đã ổn định hóa nêu trên, vốn chỉ bậc một. |

Hỗ trợ dtype: Torch `{fp16, bf16, fp32, fp64}`; Triton `{fp16, bf16, fp32}` với bf16 cần hỗ trợ phần cứng native. Dtype hỗn hợp chỉ được chấp nhận dưới autocast (giải quyết thành autocast dtype).

`is_available()` trả về `True` khi Triton + CUDA/HIP kernels có thể được chọn (không kiểm tra dtype hay policy).

---

## 8. Low-level ops

`orda_kernel.ops.tied_teacher.tied_distillation_loss`, `ops.separate_teacher.separate_distillation_loss`, và `ops.precomputed_teacher.precomputed_distillation_loss` cho phép truy cập trực tiếp các mode slice. Khác biệt so với `distillation_loss`:

- Trả về một 4-tuple thuần `(loss, student_ce, teacher_ce, kl)`.
- Mô hình thực thi chỉ Triton (không có tham số `backend`/`profile`); chúng dùng cùng planner và execution engine, nên có cùng validation, OOM retry, và chunk caching.
- `autotune` mặc định `False` (các profile cấp cao bật nó theo mặc định).
- Các API low-level cung cấp `assume_valid_labels: bool = False` cho trusted-label fast path.
- Mặc định `teacher_ce_weight=None`: tied `1.0`, separate `0.0`, precomputed `0.0`.
- `precomputed_distillation_loss` còn nhận hoặc các tensor cache thô (`logits_teacher` hoặc `teacher_hidden`+`teacher_weight`) hoặc một descriptor đã kiểm tra qua keyword `teacher=PrecomputedTeacher(...)`, cộng `assume_valid_teacher_cache: bool = False` để bỏ qua việc kiểm tra lại stamp mỗi call.

Các lớp autograd `Function` và kernel helpers dưới các hàm này là private implementation API.

---

## 9. API Aligned-span

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

- `student_hidden: (BT_s, H)`, `student_weight: (V_s, H)` (`V_s >= 1`), `student_labels: (BT_s,) torch.long`, tất cả trên một device.
- `backend="triton"` yêu cầu một `PrecomputedSpanTeacher` cộng Triton và một dtype CUDA/HIP được hỗ trợ; một `CrossVocabTeacher` luôn chạy trên Torch reference (`auto` fallback im lặng — đó là kiểu teacher reference).
- `kernel_config` nhận `grad_weight_accumulation` cộng các trường chunking/autotune/label; các trường KL tokenwise (`kl_weight`, `kl_temperature`, `kl_teacher_temperature`, `kl_divergence`, `kl_alpha`, `kl_beta`, `entropy_gate`) bị từ chối.
- `gradient_materialization_scale` và `grad_scaler` theo contract tokenwise ở trên và loại trừ lẫn nhau.
- Luôn trả về `teacher_ce = 0`; `kl` chứa binary span divergence đã rút gọn.

### `AlignedSpanPairs(student_ranges, teacher_ranges)`

Hai tensor `(C, 2)` int32/int64 chứa các range nửa mở `[start, end)`, cùng `C`, kiểm tra theo biên batch lúc call (`0 <= start < end <= rows`).

### `AlignedLikelihoodConfig`

| Trường | Mặc định | Contract |
|---|---|---|
| `student_ce_weight` | `1.0` | hữu hạn, `>= 0` |
| `span_kl_weight` | `1.0` | hữu hạn, `>= 0` |
| `ignore_index` | `-100` | int |
| `reduction` | `"mean"` | `"mean"` hoặc `"sum"` (span mean chỉ đếm các span được giữ) |
| `objective` | `"binary_kl"` | hoặc `"binary_ce"` |

### `CrossVocabTeacher(hidden, weight, labels)`

Teacher live trên vocabulary riêng (`hidden: (BT_t, D_t)`, `weight: (V_t, D_t)`, `labels: (BT_t,) torch.long`), cùng device với các tensor student. Span likelihoods tính dưới `no_grad`; teacher luôn detached. Các row bị ignore được mask trước projection; kết quả GEMM non-finite trên một row đang hoạt động bị từ chối.

### `PrecomputedSpanTeacher(span_log_probs, trust_cache_immutability=False, teacher_ranges=None, trust_teacher_ranges=False)`

Tensor float 1D chứa log-likelihood theo span: giá trị `<= 0` hoặc `-inf` (zero-support), không `NaN`/`+inf`, cấm `requires_grad`, số entry phải bằng `C`. Cùng bảo vệ validation-stamp như `PrecomputedTeacher`.

`teacher_ranges` ràng buộc cache với các range `(C, 2)` mà nó được build từ đó (`build_precomputed_span_teacher` điền vào); lúc call nó được so với `AlignedSpanPairs.teacher_ranges` và một sai khác sẽ báo lỗi. Một cache thô tự dựng phải hoặc cung cấp `teacher_ranges` hoặc chọn không dùng với `trust_teacher_ranges=True` (loại trừ lẫn nhau với `teacher_ranges`).

### `build_precomputed_span_teacher(teacher_hidden, teacher_weight, teacher_labels, teacher_ranges, *, ignore_index=-100)`

Tạo một `PrecomputedSpanTeacher` đã detach trong các chunk có giới hạn (~`2**24` phần tử logit mỗi chunk, thống kê fp32 hoặc fp64 cho input fp64). Kiểm tra shapes, dtypes, tính đồng nhất device, phạm vi label, biên range, và từ chối mọi span không có token teacher hợp lệ, báo các chỉ số span vi phạm.

---

## 10. Builders precomputed-input

Cả bốn helper là bước chuẩn bị phía Torch cho `PrecomputedTeacher`: chúng detach output và không bao giờ chọn kernel path hay ảnh hưởng autotune identity.

### `build_exact_token_map(student_vocab, teacher_vocab) -> Tensor[V_t] int64`

Khớp chuỗi chính xác từ mỗi token teacher tới một chỉ số student, `-1` khi không khớp. Các chuỗi student trùng lặp giải quyết về lần xuất hiện đầu. Mọi entry phải là `str`.

### `project_teacher_logits(teacher_logits, token_map, student_vocab_size, *, unmatched="renormalize", temperature=1.0) -> Tensor[N, V_s]`

Chiếu logits teacher `[N, V_t]` lên các cột student; các token teacher trùng lặp cộng dồn khối lượng xác suất; các cột student không khớp nhận `-inf`, nên xác suất softmax của chúng bằng chính xác không tại mọi temperature và sống sót qua một phép cast fp16 phía sau.

- `unmatched="renormalize"`: khối lượng đã khớp được rescale về một (log-probabilities đúng chuẩn).
- `unmatched="mask"`: giữ log-mass đã khớp tuyệt đối (log-sum-exp theo nhóm).
- `temperature=T` (mặc định `1.0`, hữu hạn `> 0`) chia raw teacher logits trước khi khối lượng được gom. Temperature không giao hoán với phép gom nhiều-về-một, nên output bị ràng buộc với `T`: khai báo nó với `PrecomputedTeacher(logits=..., logits_temperature=T)` để loss áp dụng `T` làm temperature teacher và không temper lần thứ hai.

Validation: `token_map` 1D int32/int64 với một entry cho mỗi cột teacher, giá trị `>= -1`, ít nhất một khớp, các chỉ số ánh xạ `< student_vocab_size`; `teacher_logits` là tensor float không có `NaN`/`+inf` (`-inf` được phép), và mỗi row cần ít nhất một logit hữu hạn trên một cột đã khớp. Các dtype half tính trong fp32. Phép chiếu ở trong log space, và các logit nguồn hữu hạn vẫn hữu hạn sau khi thu hẹp ngay cả khi xác suất của chúng không đáng kể. Output được detach.

### `mix_teacher_logits(logits_list, weights, *, mode="arithmetic") -> Tensor[N, V]`

Trộn nhiều teacher đã detach trên một vocabulary chung.

- Inputs: `>= 1` teacher, tất cả `[N, V]` với shape/device/dtype giống hệt, không `NaN`/`+inf`, không có row chết hoàn toàn. `weights`: 1D, một cho mỗi teacher, hữu hạn, nghiêm ngặt `> 0` (chuẩn hóa nội bộ trong log space).
- `mode="arithmetic"`: trung bình có trọng số của các xác suất → log-probabilities đã chuẩn hóa.
- `mode="geometric"`: log-score có trọng số kiểu product-of-experts; zero support ở bất kỳ teacher nào vẫn là zero, và mỗi row phải giữ ít nhất một token có support chung (nếu không báo `ValueError`).
- Trộn geometric và `generalized_jensen_shannon` yêu cầu mọi trọng số đã chuẩn hóa vẫn biểu diễn được trong compute dtype; trộn arithmetic giữ các trọng số cực trị trong log space.
- Compute dtype fp32 (fp64 vào → fp64); output được detach.

### `generalized_jensen_shannon(logits_list, weights) -> Tensor[N]`

Chỉ số chẩn đoán mức đồng thuận teacher theo row, đã detach (trung bình có trọng số của KL từ mỗi teacher tới hỗn hợp arithmetic), chặn dưới `>= 0`. Cùng validation input như `mix_teacher_logits`.

---

## 11. Điều khiển runtime

- `is_available() -> bool` — Triton import được và `torch.cuda.is_available()`.
- `reset_autotune_cache()` — xóa trạng thái autotune trong bộ nhớ và trên đĩa.
- `begin_autotune_run()` — bắt đầu một run process-local mới (trạng thái tuning theo run) mà không loại bỏ các winner đã lưu.
- `__version__` — phiên bản distribution, `"0+unknown"` khi ngoài một package đã cài đặt.

## 12. Biến môi trường autotune

Winners được cache trong một LRU trong bộ nhớ và một SQLite (WAL) store, khóa bởi device fingerprint, shape/pipeline workload, dtype, và một timed-source AST digest. Cache root: `ORDA_AUTOTUNE_CACHE_DIR`, ngược lại `%LOCALAPPDATA%/orda_kernel` (Windows), `$XDG_CACHE_HOME/orda_kernel`, hoặc `~/.cache/orda_kernel`. Trên cluster hãy trỏ nó tới bộ nhớ node-local — khóa SQLite/OS không đáng tin cậy trên mọi NFS; một store không dùng được sẽ cảnh báo một lần cho mỗi path và hạ xuống autotuning process-local.

Các giá trị ngoài phạm vi được clamp kèm `RuntimeWarning`; các giá trị không phải số nguyên quay về mặc định.

| Biến | Mặc định | Biên | Ý nghĩa |
|---|---|---|---|
| `ORDA_AUTOTUNE_CACHE_DIR` | thư mục cache nền tảng | — | Ghi đè vị trí store. |
| `ORDA_AUTOTUNE_MEMORY_ENTRIES` | `4096` | `[1, 65536]` | Kích thước LRU winner trong bộ nhớ. |
| `ORDA_AUTOTUNE_DISK_ENTRIES` | `4096` | `[1, 65536]` | Kích thước LRU của disk store. |
| `ORDA_AUTOTUNE_LEASE_SECONDS` | `60` | `[10, 3600]` | Thời hạn device-lease liên tiến trình. |
| `ORDA_AUTOTUNE_LEASE_POLL_MS` | `100` | `[25, 5000]` | Khoảng poll khi chờ lease. |
| `ORDA_AUTOTUNE_LEASE_WAIT_SECONDS` | `2` | `[1, 3600]` | Thời gian chờ tối đa cho một device lease đang bận; khi hết hạn call dùng cached hoặc default config gần nhất thay vì benchmark. |
| `ORDA_AUTOTUNE_SQLITE_TIMEOUT_SECONDS` | `15` | `[1, 120]` | Timeout busy của SQLite. |
| `ORDA_AUTOTUNE_SQLITE_INIT_RETRIES` | `3` | `[1, 10]` | Số lần thử khởi tạo schema. |
| `ORDA_AUTOTUNE_SQLITE_INIT_RETRY_MS` | `50` | `[1, 1000]` | Độ trễ giữa các lần thử khởi tạo. |
| `ORDA_AUTOTUNE_SCREEN_WARMUP_REPS` | `2` | `[0, 2]` | Số rep warmup screening cho mỗi candidate. |
| `ORDA_AUTOTUNE_SCREEN_REPS` | `3` | `[3, 5]` | Số rep đo screening. |
| `ORDA_AUTOTUNE_FINALISTS` | `3` | `[1, 3]` | Số candidate vào vòng đo robust. |
| `ORDA_AUTOTUNE_ROBUST_MIN_REPS` | `4` | `[1, 8]` | Số rep tối thiểu ở robust stage. |
| `ORDA_AUTOTUNE_ROBUST_MAX_REPS` | `12` | `[1, 12]` | Số rep tối đa ở robust stage. |
| `ORDA_AUTOTUNE_ROBUST_TARGET_MS` | `80` | `[1, 500]` | Ngân sách đo ở robust stage. |
| `ORDA_AUTOTUNE_TIMING_BATCH` | `1` | `[1, 16]` | Số launch cho mỗi mẫu đo. |
| `ORDA_AUTOTUNE_FAILURE_BACKOFF_MS` | `1000` | `[10, 60000]` | Backoff cơ sở sau một lỗi benchmark tạm thời. |
| `ORDA_AUTOTUNE_FAILURE_BACKOFF_MAX_MS` | `30000` | `[10, 600000]` | Trần backoff (không bao giờ dưới mức cơ sở). |

## 13. Tổng hợp exception

| Exception | Được báo cho |
|---|---|
| `ValueError` | Hầu hết mọi vi phạm contract: shapes, dtypes, devices, phạm vi weight/temperature/hyperparameter, phạm vi label, bất biến descriptor, cache mutation, `mutual`/`cojeffreys` với `PrecomputedTeacher`, precomputed teacher CE trên `backend="triton"`, xung đột chunk request. |
| `TypeError` | Sai loại đối tượng: các trường descriptor không phải tensor, chunk request không phải `ChunkSize`, gate không phải `EntropyGate`, tham số không phải `AlignedSpanPairs`/`AlignedLikelihoodConfig`. |
| `UnsupportedTritonPolicyError` (lớp con của `ValueError`) | `backend="triton"` với một policy hợp lệ về ngữ nghĩa mà không Triton specialization nào thực thi. Không export ở package root — hãy bắt `ValueError`; các trường hợp phổ biến (precomputed teacher CE, `CrossVocabTeacher` live) được chặn trước bằng các thông báo `ValueError` thường. |
| `RuntimeError` | `backend="triton"` khi không có Triton/CUDA-HIP; OOM khi phân hoạch tự động không chia nhỏ thêm được (thông báo bao gồm shape, phân hoạch đã dùng, và cách khắc phục theo `num_chunks`); bất nhất nội bộ của plan. |
| `RuntimeWarning` | Thông báo fallback `auto`→Torch một lần; các suy giảm của autotune store/môi trường. |
