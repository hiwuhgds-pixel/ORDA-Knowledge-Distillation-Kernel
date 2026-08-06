# Các biến thể thực thi

Tài liệu này liệt kê các biến thể package thực thi: cách một call định tuyến tới một trong tám kernel path, mỗi path hỗ trợ những gì, contract của cả mười lăm biến thể KL divergence, và các giới hạn xuyên suốt. Tài liệu đồng hành: [reference.md](reference.md) (contract API theo tham số) và [architecture.md](architecture.md) (tổ chức nội bộ).

Ký hiệu: `p` là phân phối student, `q` là phân phối teacher, `BT` là số dòng đã flatten `batch * seq`, `H` là hidden size của student, `D_t` là hidden size của teacher, `V` là vocabulary size. `T_s` là `kl_temperature`, `T_t` là `kl_teacher_temperature`.

---

## 1. Cách một call định tuyến biến thể

Routing mang tính xác định và suy ra từ ba đầu vào: kiểu teacher descriptor, các loss weight, và biến thể KL.

### 1.1 Teacher descriptor → teacher mode

| Descriptor | Teacher mode |
|---|---|
| `TiedTeacher(hidden, mode=None)` | `tied` |
| `SeparateTeacher(hidden, weight)` | `separate` |
| `PrecomputedTeacher(logits=… \| teacher_hidden=…, teacher_weight=…)` | `precomputed` |
| `CrossVocabTeacher` / `PrecomputedSpanTeacher` (chỉ aligned-span API) | `precomputed` (aligned-span path) |

### 1.2 Teacher mode + weights + variant → kernel path

Đánh giá theo thứ tự sau:

1. `kl_divergence ∈ {"mutual", "cojeffreys"}` và `kl_weight != 0` → cotrained path cho mode đó.
2. Ngược lại `teacher_ce_weight > 0` → trained path.
3. Ngược lại → frozen path.
4. Mode `precomputed` luôn giải quyết thành `precomputed_tokenwise`.

`teacher_ce_weight=None` (mặc định) giải quyết theo mode trước khi routing:

| Mode | `teacher_ce_weight` mặc định |
|---|---|
| `tied` | `1.0` |
| `separate` | `0.0` |
| `precomputed` | `0.0` |

Cùng các mặc định này áp dụng cho các low-level op (`tied_distillation_loss`, `separate_distillation_loss`, `precomputed_distillation_loss`).

### 1.3 Kernel path → chính sách gradient teacher

| Kernel path | Chính sách gradient teacher |
|---|---|
| `tied_frozen`, `separate_frozen`, `precomputed_tokenwise` | `none` |
| `tied_trained`, `separate_trained` | `ce` (chỉ teacher CE) |
| `tied_cotrained`, `separate_cotrained` | `kl`, hoặc `ce_kl` khi `teacher_ce_weight != 0` |
| `precomputed_aligned_span` | `none` (chỉ student hidden và weight) |

`TiedTeacher.mode` khai báo tường minh không bao giờ thay đổi routing; nó khẳng định rằng vai trò đã giải quyết khớp với khai báo (mục 2.1).

---

## 2. Các tokenwise kernel path

Cả bảy tokenwise path chia sẻ contract shape `student_hidden: (BT, H)`, `weight: (V, H)`, `labels: (BT,) torch.long`, mọi tensor trên một device, cùng các giới hạn chung ở mục 5.

### 2.1 `tied_frozen`

- **Chọn khi:** `TiedTeacher`, `teacher_ce_weight == 0`, không có KL `mutual`/`cojeffreys` hoạt động.
- **Tensors teacher:** `hidden: (BT, H)` — phải khớp shape student chính xác; projection `weight` được chia sẻ.
- **Gradients:** chỉ student hidden và shared weight. Teacher hidden có thể `require_grad`, nhưng path này không đóng góp gradient cho nó.
- **Biến thể KL hoạt động:** cả mười ba biến thể teacher-detached. `mutual`/`cojeffreys` với `kl_weight != 0` định tuyến lại sang `tied_cotrained`.
- **KL không hoạt động (`kl_weight == 0`):** chấp nhận bất kỳ biến thể nào trong mười lăm; plan canonicalize biến thể thành `forward` và cả hai temperature thành 1.
- **Kiểm tra mode khai báo:** `TiedTeacher(mode="frozen")` báo `ValueError` khi `teacher_ce_weight != 0` hoặc khi cấu hình KL `mutual`/`cojeffreys` hoạt động.

### 2.2 `tied_trained`

- **Chọn khi:** `TiedTeacher` và `teacher_ce_weight > 0` (tường minh, hoặc qua mặc định tied `1.0`), không có `mutual`/`cojeffreys` hoạt động.
- **Gradients:** teacher CE chảy tới teacher hidden và shared weight, ngoài mọi gradient student. Term KL giữ teacher detached.
- **Kiểm tra mode khai báo:** `TiedTeacher(mode="trained")` yêu cầu `teacher_ce_weight > 0` và không có biến thể teacher-KL-gradient hoạt động.

### 2.3 `tied_cotrained`

- **Chọn khi:** `TiedTeacher`, `kl_divergence ∈ {"mutual", "cojeffreys"}`, `kl_weight != 0`.
- **Biến thể KL hoạt động:** chỉ `mutual` và `cojeffreys`. Không có dạng không hoạt động — với `kl_weight == 0` call định tuyến sang frozen/trained.
- **Gradients:** teacher hidden nhận gradient KL (cộng gradient CE khi `teacher_ce_weight != 0`), và shared projection weight tích lũy gradient từ cả nửa student lẫn nửa teacher của stacked logits buffer.
- **Giới hạn:** `kl_teacher_temperature` phải bằng `kl_temperature`; temperature bất đối xứng chỉ xác định cho các biến thể teacher-detached.
- **Kiểm tra mode khai báo:** `TiedTeacher(mode="cotrained")` yêu cầu `kl_weight > 0` với `kl_divergence ∈ ("mutual", "cojeffreys")`.

### 2.4 `separate_frozen`

- **Chọn khi:** `SeparateTeacher`, `teacher_ce_weight == 0` (mặc định separate), không có `mutual`/`cojeffreys` hoạt động.
- **Tensors teacher:** `hidden: (BT, D_t)`, `weight: (V, D_t)`. `D_t` có thể khác `H`; `V` phải bằng vocabulary của student.
- **Gradients:** không gradient nào tới teacher; teacher logits tạo theo chunk và tiêu thụ ở dạng detached.
- **Biến thể KL hoạt động/không hoạt động:** cùng tập với `tied_frozen`.

### 2.5 `separate_trained`

- **Chọn khi:** `SeparateTeacher` và `teacher_ce_weight > 0`.
- **Gradients:** teacher CE sở hữu gradient teacher — chảy tới `teacher.hidden` / `teacher.weight` nào có `requires_grad=True`. Mọi biến thể KL teacher-detached không gửi gradient KL tới teacher.

### 2.6 `separate_cotrained`

- **Chọn khi:** `SeparateTeacher`, `kl_divergence ∈ {"mutual", "cojeffreys"}`, `kl_weight != 0`.
- **Gradients:** KL `mutual`/`cojeffreys` sở hữu gradient teacher, cộng đóng góp teacher CE tùy chọn (chính sách `ce_kl`).
- **Giới hạn:** yêu cầu temperature bằng nhau, như với `tied_cotrained`.

### 2.7 `precomputed_tokenwise`

- **Chọn khi:** teacher là `PrecomputedTeacher`, luôn luôn.
- **Biểu diễn (chính xác một):**
  - `logits: (BT, V)` — teacher logits đã cache. Giá trị phải hữu hạn hoặc `-inf` (zero support; `NaN`/`+inf` bị từ chối lúc construction). Dtype có thể khác model dtype; cached logits được row-center trong dtype nguồn, sau đó chuyển đổi theo chunk. Underflow hữu hạn được clamp vào phạm vi compute trong khi `-inf` giữ chính xác.
  - `teacher_hidden: (BT, D_t)` + `teacher_weight: (V, D_t)` — tính lại teacher logits từ trunk output đã cache. Giá trị phải hữu hạn nghiêm ngặt; kết quả GEMM non-finite hiện ra như một row số học không hợp lệ.
- **Contract gradient:** cached tensors không được `require_grad` (`ValueError` lúc construction). Không gradient nào tới precomputed teacher; không tồn tại trained/cotrained precomputed path.
- **Bảo vệ cache-mutation:** descriptor chụp một validation stamp (object id, data pointer, shape, stride, storage offset, dtype, device, layout, `requires_grad`, version counter) lúc construction và kiểm tra lại ở mỗi call. In-place mutation sau construction báo `ValueError`; hãy tạo descriptor mới. `trust_cache_immutability=True` cần thiết bất cứ khi nào `_version` không khả dụng, bao gồm inference tensors và tensor subclasses không có counter đọc được. `.data` mutation không an toàn nằm ngoài bảo đảm trong cả hai trường hợp.
- **Biến thể KL:** mười ba biến thể teacher-detached. `mutual`/`cojeffreys` bị từ chối với `ValueError` ngay cả khi `kl_weight == 0` — precomputed mode không bao giờ trả gradient KL cho teacher.
- **Teacher CE:** không hỗ trợ trên Triton path.
  - `backend="triton"` + `teacher_ce_weight != 0` → `ValueError`.
  - `backend="auto"` + `teacher_ce_weight != 0` → fallback về Torch reference với cảnh báo fallback một lần.
- **Dead rows:** row có cached logits toàn `-inf` được coi là zero-support teacher row: teacher CE và KL ghim về `0` (vẫn tính trong mẫu số `mean`).

---

## 3. Aligned-span path (`precomputed_aligned_span`)

`aligned_span_distillation_loss` là một objective riêng cho cross-tokenizer distillation: thay vì KL theo cột trên vocabulary chung, mỗi span căn chỉnh so sánh log-likelihood mà hai model gán cho cùng một đoạn văn bản. Không gian sự kiện theo span là nhị phân — phần tiếp nối quan sát được so với bất kỳ phần nào khác — nên objective là binary KL hoặc binary CE giữa các span log-likelihood. Cố ý không phải là một `KLVariant`.

**Objectives:** `binary_kl` (mặc định) hoặc `binary_ce`, chọn bởi `AlignedLikelihoodConfig.objective`. Cả hai chia sẻ cùng fused pipeline và một autotune winner.

**Teachers:**

| Teacher | Backend | Ghi chú |
|---|---|---|
| `CrossVocabTeacher(hidden, weight, labels)` | chỉ Torch reference | Live teacher trên vocabulary riêng; luôn detached. `backend="triton"` báo `ValueError`; `backend="auto"` dùng reference. |
| `PrecomputedSpanTeacher(span_log_probs, *, teacher_ranges=None, trust_teacher_ranges=False)` | Triton (hoặc Torch) | Cached per-span log-likelihoods `[C]`; giá trị phải `<= 0` hoặc `-inf` (`NaN`/`+inf` bị từ chối); không được `require_grad`; cùng bảo vệ validation-stamp như `PrecomputedTeacher`; số entry phải bằng số span. `teacher_ranges` ràng buộc cache với các range đã build (kiểm tra với `AlignedSpanPairs.teacher_ranges` lúc call); một cache thô phải đặt nó hoặc `trust_teacher_ranges=True`. |

**Span pairs:** `AlignedSpanPairs(student_ranges, teacher_ranges)` — hai tensor `(C, 2)` int32/int64 chứa các range token nửa mở `[start, end)`, một teacher range cho mỗi student range. Lúc call mọi range phải thỏa `0 <= start < end <= rows` cho phía của nó.

**Các trường hợp và giới hạn:**

- Span không có token student hợp lệ (non-ignored) không mang bằng chứng: loại khỏi cả loss lẫn mẫu số `mean`.
- Với `CrossVocabTeacher` live, span teacher bị ignore toàn bộ cũng bị loại. Với precomputed teacher chỉ phía student kiểm soát span — `-inf` đã cache là một giá trị zero-support riêng biệt và hợp lệ.
- `build_precomputed_span_teacher` từ chối mã hóa span không có bằng chứng: mỗi teacher span phải chứa ít nhất một token teacher không bị ignore, và lỗi báo các chỉ số span vi phạm.
- Các trường `KernelConfig` chỉ dành cho tokenwise KL (`kl_weight`, `kl_temperature`, `kl_teacher_temperature`, `kl_divergence`, `kl_alpha`, `kl_beta`, `entropy_gate`) bị từ chối với `ValueError` khi truyền qua `kernel_config`.
- Triton pipeline chạy hai stage: stage A lưu NLL theo row và log-sum-exp; các phép rút gọn FP64 prefix/difference tạo span losses và một hệ số vô hướng cho mỗi token row; stage B chuyển hệ số cộng LSE đã lưu thành gradients logits `a_i * (p - onehot)`. Các lần chạy một chunk tái sử dụng logits buffer của stage A; các lần chạy nhiều chunk tính lại chunk-local logits.
- Student span log-likelihood bị chặn tại `-1e-7` trong forward objective. Đây là một surrogate bị chặn: backward dùng đạo hàm tại điểm chặn (straight-through), và span được giữ vẫn nằm trong mẫu số mean.
- FP64 workspace cố định là `O(BT + spans)` và không giảm khi tăng số chunk. OOM của nó được báo trực tiếp thay vì kích hoạt vòng lặp chunk retry (chỉ chunk-local vocabulary OOM mới retry được).
- Path này không tính tokenwise KL và không tính teacher CE; thành phần `teacher_ce` trả về luôn là `0`, và `out.kl` chứa binary span divergence đã rút gọn.

---

## 4. Các biến thể KL divergence

`kl_divergence` chọn một trong mười lăm giá trị. Quy tắc chung:

- Phân phối student là `softmax(student_logits / T_s)`; phân phối teacher là `softmax(teacher_logits / T_t)`. Mỗi objective theo row được nhân với `T_s**2`.
- **Các biến thể teacher-detached** (mọi biến thể trừ `mutual`, `cojeffreys`) không bao giờ gửi gradient KL tới teacher và hỗ trợ temperature bất đối xứng (`kl_teacher_temperature != kl_temperature`).
- `mutual` và `cojeffreys` yêu cầu `SeparateTeacher` hoặc `TiedTeacher`, temperature teacher bằng nhau, và định tuyến tới các cotrained path.
- Vì xác suất student đến từ softmax trên các logit hữu hạn, `p > 0` luôn đúng. Zero teacher support (`q == 0`) có thể xảy ra qua `PrecomputedTeacher` với logits `-inf` (ví dụ token-map projection). Hành vi tại zero support được xác định cho từng biến thể bên dưới.
- `kl_alpha` / `kl_beta` được kiểm tra bất cứ khi nào được cung cấp. Khi biến thể đã chọn không đọc một tham số, giá trị cung cấp vẫn phải là float hữu hạn trong khoảng mở `(0, 1)`. Khi term KL không hoạt động, phạm vi riêng của biến thể đã chọn được dùng để kiểm tra; plan sau đó canonicalize giá trị thành `0.5` cho mục đích identity.
- Với `kl_weight == 0` không tính KL, biến thể canonicalize thành `forward`, và cả hai temperature ghim về `1.0` trên plan.

| # | Biến thể | Objective (theo row, trước hệ số `T_s**2`) | `kl_alpha` | `kl_beta` | Grad teacher | Hành vi zero-support |
|---|---|---|---|---|---|---|
| 0 | `forward` | `KL(q‖p) = Σ q·(log q − log p)` | — | — | detached | cột `q == 0` đóng góp chính xác `0` |
| 1 | `jeffreys` | `KL(q‖p) + KL(p‖q)` | — | — | detached | nửa reverse: bất kỳ `p > 0` nơi `q == 0` → row `+inf`, gradient bằng không |
| 2 | `mutual` | `Σ q̄·(log q̄ − log p) + Σ p̄·(log p̄ − log q)` (dấu gạch = target detached) | — | — | **gradient KL** | softmax teacher ⇒ `q > 0`; không có trường hợp đặc biệt |
| 3 | `cojeffreys` | Jeffreys đối xứng với cả hai phía live | — | — | **gradient KL** | softmax teacher ⇒ `q > 0`; không có trường hợp đặc biệt |
| 4 | `reverse` | `KL(p‖q) = Σ p·(log p − log q)` | — | — | detached | `p > 0` nơi `q == 0` → row `+inf`, gradient bằng không |
| 5 | `alpha_divergence` | `(1 − Σ p^α q^(1−α)) / (α(1−α))` | `α ∈ (0, 1)`, mặc định `0.5` | — | detached | cột `q == 0` rơi khỏi tổng affinity |
| 6 | `skew_forward` | `KL(q ‖ m)`, `m = (1−β)q + βp` | — | `β ∈ (0, 1)`, mặc định `0.5` | detached | hữu hạn theo cấu tạo; chặn trên bởi `−log(1−β)` mỗi row |
| 7 | `skew_js` | `(1−β)·KL(q‖m) + β·KL(p‖m)` | — | `β ∈ (0, 1)`, mặc định `0.5` | detached | hữu hạn theo cấu tạo |
| 8 | `entropy_adaptive` | `α(h)·KL(q‖p) + (1−α(h))·KL(p‖q)` với `α(h) = α_min + (α_max−α_min)·σ((h−τ)/scale)`, `h = H(q)/log V` (`0` khi `V == 1`) | — (dùng `entropy_gate`) | — | detached; gate cũng detached | nửa reverse: thiếu support → row `+inf`, gradient bằng không |
| 9 | `skew_reverse` | `KL(p ‖ m)`, `m = (1−β)q + βp` | — | `β ∈ (0, 1)`, mặc định `0.5` | detached | hữu hạn theo cấu tạo; chặn trên bởi `−log β` mỗi row |
| 10 | `renyi` | Rényi teacher-first `D_r(q‖p) = log(Σ q^r p^(1−r)) / (r−1)` | `r ∈ (0, 2]`, mặc định `0.5`, loại trừ `\|r−1\| < 0.01` | — | detached | cột có `p = q = 0` bị bỏ qua; row bất ổn/không có support → `+inf` với gradient bằng không; input non-finite khác `-inf` → `NaN` |
| 11 | `renyi_reverse` | Rényi student-first `D_r(p‖q)` | như `renyi` | — | detached | cùng sentinel với `renyi` |
| 12 | `dkd` | `λ_t·TCKD + λ_n·NCKD` (tách target/non-target) | `λ_t ∈ (0, 16]`, mặc định `1` | `λ_n ∈ (0, 16]`, mặc định `1` | detached | khối lượng non-target của teacher trên support student thiếu → row `+inf`, gradient KL bằng không; CE không đổi |
| 13 | `abkd` | AB-divergence với tham số `(a, b)` | `a ∈ [0.05, 2]`, mặc định `0.5` | `b ∈ [0.05, 2]`, mặc định `0.5` | detached | cột `p = q = 0` đóng góp `0`; input non-finite khác `-inf` → `NaN` với gradient bằng không |
| 14 | `triangular` | phân biệt tam giác `Σ (p−q)²/(p+q)`, đối xứng | — | — | detached | cột `p + q == 0` bị bỏ qua; chặn trên bởi `2` mỗi row |

Ghi chú theo biến thể:

- **`mutual` so với `cojeffreys`:** `mutual` theo hướng deep mutual learning — mỗi hướng coi phân phối kia là target detached, nên student học từ `KL(q̄‖p)` và teacher học từ `KL(p̄‖q)`. `cojeffreys` giữ cả hai nửa live cho cả hai phía (objective co-training Jeffreys đối xứng). Cả hai từ chối `PrecomputedTeacher`.
- **`renyi` / `renyi_reverse`:** tại `r = 1` cả hai giới hạn là KL thông thường — dùng `forward` hoặc `reverse` ở đó; dải bảo vệ `|r−1| >= 0.01` thực thi điều này. Gần identity (`|r−1| <= 0.125`) reference chuyển sang đánh giá `log1p` không có triệt tiêu với một control variate.
- **`dkd`:** TCKD là binary KL giữa các cặp khối lượng `(target, non-target)`; NCKD là forward KL giữa các phân phối non-target. Đây là biến thể duy nhất đọc `labels` bên trong term KL, nên các row bị ignore không đóng góp KL. TCKD vẫn hoạt động tại cả hai điểm mút teacher-target; NCKD bằng không chỉ khi teacher không có non-target support. Paper DKD thường dùng `λ_n ≈ 8`; mặc định ở đây là `(1, 1)` trung tính — đặt `kl_beta=8` cho trọng số theo paper.
- **`entropy_adaptive`:** gate tính từ phân phối teacher đã detach và không bao giờ nhận gradient. Mặc định: `EntropyGate(alpha_min=0.0, alpha_max=1.0, tau=0.5, scale=0.1)` với ràng buộc `0 <= alpha_min <= alpha_max <= 1`; `scale` và `1 / scale` phải là giá trị float32 normal hữu hạn. `entropy_gate` cung cấp cho biến thể khác được chấp nhận (kiểm tra kiểu) nhưng bị canonicalize bỏ đi.

### 4.1 Tương thích mode × variant (KL hoạt động, `kl_weight != 0`)

| Nhóm biến thể | `TiedTeacher` | `SeparateTeacher` | `PrecomputedTeacher` |
|---|---|---|---|
| 13 biến thể teacher-detached | ✓ (frozen/trained paths) | ✓ (frozen/trained paths) | ✓ |
| `mutual`, `cojeffreys` | ✓ (cotrained path) | ✓ (cotrained path) | ✗ `ValueError` (kể cả khi không hoạt động) |

Một teacher được train bằng CE kết hợp với biến thể KL detached chỉ nhận gradient CE của nó.

### 4.2 Giới hạn temperature

- `kl_temperature` và `kl_teacher_temperature` phải hữu hạn và `> 0`.
- `1/T_s` và `T_s**2` phải biểu diễn được dưới dạng giá trị float32 normal, và `1/T_t` cũng vậy — temperature cực nhỏ hoặc cực lớn báo `ValueError` thay vì âm thầm flush về zero/inf trong kernel.
- `kl_teacher_temperature=None` (mặc định) nghĩa là `T_t = T_s`.
- Cả hai temperature bằng `1.0` chọn một fast path chuyên biệt (`t_is_one`) bỏ qua hoàn toàn scaling.
- Kiểm tra temperature chạy cả khi term KL không hoạt động; plan sau đó ghim cả hai temperature về `1.0`.

---

## 5. Các trường hợp và giới hạn xuyên suốt

### 5.1 Backend × device × dtype

| Backend | Device | Model dtype hỗ trợ | Ghi chú |
|---|---|---|---|
| `torch` | CPU hoặc CUDA/HIP | `float16`, `bfloat16`, `float32`, `float64` | Implementation reference; tạo toàn bộ logits `(BT, V)`. |
| `triton` | chỉ CUDA/HIP | `float16`, `bfloat16`, `float32` | `bfloat16` cần hỗ trợ phần cứng native (không emulation); CUDA trước Ampere từ chối nó. |
| `auto` | bất kỳ | hợp của các mục trên | Dùng Triton khi khả dụng và policy được hỗ trợ; ngược lại fallback về Torch với `RuntimeWarning` một lần trên toàn tiến trình. |

- Mọi model tensor (`student_hidden`, `weight`, các tensor teacher đang hoạt động) phải chia sẻ một dtype khi ngoài autocast. Trong ngữ cảnh autocast đang hoạt động, một hỗn hợp được hỗ trợ sẽ giải quyết thành autocast dtype; bfloat16 autocast trên device không có BF16 native sẽ fallback về dtype của tensor thay vì ép kiểu.
- `PrecomputedTeacher.logits` được miễn quy tắc đồng nhất dtype: cached logits chuyển đổi theo chunk sang compute dtype.
- `float64` chỉ dành cho Torch. `backend="triton"` với `float64` báo lỗi.
- `backend="triton"` báo `RuntimeError` khi không có Triton + CUDA/HIP tensors và `ValueError` cho dtype không được hỗ trợ. Các policy không hỗ trợ đã biết bị từ chối ngay từ đầu với `ValueError` nêu rõ cách sửa (precomputed + teacher CE, `CrossVocabTeacher` live trên aligned-span path); `UnsupportedTritonPolicyError` (một lớp con của `ValueError`) là backstop lúc planning cho mọi policy hợp lệ về ngữ nghĩa còn lại mà không Triton specialization nào thực thi.
- Mọi entry point public đều `torch._dynamo.disable`d: một caller đã compile sẽ thấy graph break tại đây, và kernel tự quản lý fusion của nó.

### 5.2 Labels và `ignore_index`

- `labels` phải là `torch.long`, shape `(BT,)`, trên device của student.
- Giá trị hợp lệ: `[0, V)` hoặc chính xác `ignore_index`. Kiểm tra chặt trên host là mặc định và báo tối đa năm giá trị vi phạm.
- `KernelConfig(assume_valid_labels=True)` bỏ qua bước quét phạm vi đồng bộ phía host — dùng khi input pipeline bảo đảm tính hợp lệ. Triton kernel vẫn giữ bảo vệ indexing an toàn với `ignore_index` ngay cả trong trusted mode: một label bị ignore hoặc ngoài phạm vi được clamp về index an toàn trước mọi truy cập bộ nhớ, nên trusted mode thay đổi chi phí đồng bộ, không phải an toàn bộ nhớ.
- `ignore_index` có thể là bất kỳ int Python nào (mặc định `-100`); nó không bao giờ được dùng làm memory index.
- Các row bị ignore được mask trước các phép GEMM, nên giá trị hidden-state non-finite trong các row bị ignore không thể làm hỏng loss hay weight gradients.

### 5.3 Ngữ nghĩa reduction

- `reduction="mean"` chia mọi thành phần cho số label không bị ignore, chặn dưới ở một; `"sum"` trả về tổng chưa chuẩn hóa.
- Các row bị ghim về zero (dead precomputed teacher rows, span bị loại) vẫn tính trong mẫu số `mean` tokenwise; các span bị loại không tính trong mẫu số span-mean.

### 5.4 Input suy biến

- **`BT == 0`:** trả về ngay với output giá trị zero mà `loss` vẫn nối tới mọi nguồn gradient live (backward hoạt động và tạo gradient bằng không).
- **Mọi weight bằng zero** (`student_ce_weight == teacher_ce_weight == kl_weight == 0`): một loss zero khả vi; không tính gì.
- **`V` phải `>= 1`**; vocabulary rỗng báo lỗi.
- **Báo cáo thành phần:** thành phần có weight `0` không bao giờ được tính và đọc chính xác `0` trong output — gán cho weight một giá trị dương để xem giá trị thật của thành phần.
- **Output dtype:** `promote_types(input_dtype, float32)` — input fp16/bf16 cho ra thành phần loss float32; float64 giữ float64 (chỉ Torch backend).

### 5.5 Vòng đời autograd

- Mỗi fused Triton forward hỗ trợ chính xác một backward pass. Các buffer gradient đã cache được scale tại chỗ trong backward (tránh sao chép toàn kích thước), điều này tiêu thụ chúng; backward lặp lại báo lỗi ngay cả khi call đầu dùng `retain_graph=True`. Hãy tính lại loss.
- Torch backend không có giới hạn một backward. Các proxy ổn định hóa `alpha_divergence`, `renyi`, `renyi_reverse`, và `abkd` chỉ bậc một; các yêu cầu bậc cao hơn báo lỗi thay vì bỏ qua độ cong KL.
- Các nhánh objective không hoạt động trả về gradient `None` cho các input không thuộc graph.

### 5.6 Chunking và bộ nhớ

- Triton backend chỉ tạo chunk-local logits; bộ nhớ vocabulary đỉnh là `O(chunk_rows * V)`. Torch reference tạo toàn bộ buffer `(BT, V)`.
- `ChunkSize.nrows(n)` (số row cố định mỗi chunk, `n >= 1`, clamp về `BT`), `ChunkSize.fast()` (phân hoạch hướng thông lượng), và `ChunkSize.dynamic()` (heuristic theo shape workload) là ba loại chunk request. Int thô, string, `None`, và các sentinel cũ bị từ chối.
- `num_chunks` loại trừ lẫn nhau với `ChunkSize.nrows`, và ghi đè `ChunkSize.fast()` cùng `ChunkSize.dynamic()`; số chunk không rỗng thực tế không được vượt `max_chunks`.
- Dynamic plans bắt đầu từ heuristic áp lực (`(BT/1024)·(V/32768)`, chặn dưới bởi `BT/4096`, làm tròn về lũy thừa của hai) bị chặn bởi `min(32, 2·max(1, BT//512))` trừ khi `max_chunks` ghi đè.
- Fast plans dùng `ceil(BT/1024)` chunks, tối thiểu 2 và tối đa 16 (trừ khi `max_chunks` ghi đè trần), bỏ qua `V`. Vượt `16·1024` rows các chunk nở rộng thay vì nhân lên, đổi bộ nhớ để giữ số launch thấp.
- Khi gặp CUDA/HIP OOM có thể retry, dynamic plans giải phóng cached allocator blocks, nhân đôi số chunk, và retry đến `max_chunks`; thông báo lỗi cuối báo shape cùng phân hoạch nó đã dùng, và đề xuất `num_chunks`. Phân hoạch thành công được cache (bounded LRU, 1024 entry) khóa bởi định danh tensor (shape/stride/dtype/device/`requires_grad`/contiguity) và specialization của plan, nên các bước sau bỏ qua các lần thử thất bại.
- Fixed plans (`ChunkSize.nrows`/`num_chunks`) và fast plans không retry. Cả hai chế độ tự động đều báo lại phân hoạch đã dùng; dưới trần chúng đề xuất phân hoạch kế tiếp làm `num_chunks`, còn tại trần chúng báo phần chunkable buffer đã cắt được và mức thu thêm nếu chia nhỏ tiếp. Một `num_chunks` tường minh triệt tiêu thông báo OOM.
- `max_fused_size` phải là lũy thừa của hai `<= 32768`; các profile chọn `16384` (`balanced`), `32768` (`fast`), `32768` (`debug`).

### 5.7 Tương tác với autotune

- KL hyperparameters là runtime scalars, nên quét `kl_beta`, `entropy_gate`, và các giá trị `kl_alpha` hay temperature thông thường không benchmark lại. Có hai ranh giới: cả hai temperature bằng chính xác `1.0` bật `t_is_one`, và `renyi` / `renyi_reverse` hoạt động với `|kl_alpha - 1| <= 0.125` chọn pipeline `renyi-near-one`. Vượt qua một trong hai ranh giới thay đổi autotune identity và có thể benchmark một winner riêng.
- Các trục specialization thay đổi autotune identity: kernel path, các cờ objective đang hoạt động (`compute_kl`, `compute_teacher_ce`), `t_is_one`, pipeline Rényi near-one, KL style, dtype, và shape workload.
- Aligned-span giữ một pipeline identity riêng; `binary_kl` và `binary_ce` chia sẻ một winner.

Xem [reference.md §12](reference.md#12-biến-môi-trường-autotune) cho bảng biến môi trường và cấu trúc cache.
