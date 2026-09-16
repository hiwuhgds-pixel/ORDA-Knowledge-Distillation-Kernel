# Hướng dẫn cho người phát triển kernel

Tài liệu này dành cho những ai thay đổi kernel: thêm một biến thể KL, một teacher mode, một launch pipeline, hoặc chỉnh sửa Triton kernel chuẩn. Nó ghi lại các nguyên tắc thiết kế, các contract không được lệch, và checklist khi mở rộng. Về hành vi runtime xem [architecture.md](architecture.md); về contract dành cho người dùng xem [reference.md](reference.md) và [variants.md](variants.md).

---

## 1. Nguyên tắc thiết kế

1. **Một thân số học, nhiều vertical slice.** Chỉ một Triton kernel được launch (`ops/_kernels/ce_kl_fwdbwd.py::ce_kl_fwdbwd_kernel`); `_kl_math.py` chứa các helper inline `@triton.jit` chỉ gọi từ bên trong kernel này, không bao giờ launch (`test_source_contracts.py` cố định quy tắc single-launch). Mỗi teacher mode giữ một vertical slice — benchmark task, vòng lặp chunk, các phép GEMM, autograd `Function`, và gradient ABI cố định nằm cùng nhau trong một mode file. Sự trùng lặp giữa các `Function` và benchmark builder theo mode là trùng lặp có chủ đích: nó bảo vệ quyền sở hữu gradient cố định, các slice dễ đọc, và các launch pipeline chuyên biệt.
2. **Tập trung hóa những gì không được phép fork.** Mã số học CE/KL, khởi tạo launch request, validation, và execution planning được quản lý tại một nguồn duy nhất. Một lỗi sửa trong `_kl_math.py` hoặc `_torch_backend.py` được sửa cho mọi mode.
3. **Policy là cold-path Python; plans bất biến.** Mọi quyết định routing diễn ra một lần cho mỗi call tại `_planning/`; execution nhận các đối tượng `ExecutionPlan` / `ChunkAttempt` đã đóng băng và không bao giờ giải quyết lại policy.
4. **Capability registry là ranh giới sản phẩm.** Kernel chuẩn có thể compile nhiều kết hợp constexpr hơn mức sản phẩm hỗ trợ. `_planning/capabilities.py` quyết định những gì tiếp cận được; mọi thứ còn lại bị từ chối trước khi launch.
5. **Torch reference là nguồn chuẩn.** Mọi hành vi kernel — bao gồm việc cố định các trường hợp biên (dead teacher rows, zero-support sentinels) — phải được phản ánh trong `_torch_backend.py`, vì các test xác minh Triton dựa trên đó và người dùng chỉ có CPU chạy trực tiếp file này.

## 2. Sơ đồ các lớp

```text
api.py / ops/{tied,separate,precomputed}_teacher.py / ops/precomputed_inputs/aligned_span.py
        │   validation (_validation.py, _scalar_validation.py), hợp nhất config (_config.py)
        ▼
_planning/            policy.py: weights+variant → kernel path + specialization (immutable plan)
        │             capabilities.py: ranh giới sản phẩm; models.py: vocabulary + hyperparam specs
        ▼
_execution/           dispatcher.py: vòng lặp chunk, OOM retry, partition cache
        │             aligned_span.py: span executor hai giai đoạn
        ▼
ops/_kernels/<mode>/  vertical slice: GEMMs, launches, autograd Function, gradient ABI
        ▼
ops/_kernels/ce_kl_fwdbwd.py   Triton kernel duy nhất được launch (+ inline _kl_math.py @triton.jit helpers)

_autotune/            full-batch benchmark, winner selection, SQLite/WAL store, leases
```

## 3. Contract của kernel chuẩn

### 3.1 Constexpr axes

Các trục specialization ngữ nghĩa của `ce_kl_fwdbwd_kernel`:

| Trục | Ý nghĩa |
|---|---|
| `TIED_LAYOUT` | Rows của student và teacher xếp chồng trong một buffer so với các buffer độc lập. |
| `COMPUTE_TEACHER_CE` | Term teacher CE hoạt động. |
| `COMPUTE_KL` | Term KL hoạt động. |
| `TEACHER_KL_GRAD` | Teacher nhận gradient KL (chỉ `mutual`/`cojeffreys`). |
| `BOTH_T_ONE` | Cả hai temperature đã giải quyết đều là `1.0` (bỏ qua scaling). |
| `KL_STYLE` | Một trong 15 mã KL dưới đây. |
| `RENYI_NEAR_ONE` | Bậc Rényi nằm trong khoảng `0.125` của một, cho phép control-variate sweep ổn định. |
| `KERNEL_STAGE` | `TOKENWISE=0`, `ALIGNED_STATS=1`, `ALIGNED_GRAD=2`. |
| `BLOCK_SIZE` | Chiều rộng launch tile (được autotune). |

Mã KL style (`kl_style_of`): `forward=0`, `jeffreys=1`, `mutual=2`, `cojeffreys=3`, `reverse=4`, `alpha_divergence=5`, `skew_forward=6`, `skew_js=7`, `entropy_adaptive=8`, `skew_reverse=9`, `renyi=10`, `renyi_reverse=11`, `dkd=12`, `abkd=13`, `triangular=14`.

Product policy tiếp cận được (15 style trên ma trận mode) tạo ra 170 legacy kernel identities. Kernel chuẩn, bao gồm hai aligned stage, rút gọn xuống 165 semantic combinations — `test_kernel_specializations.py` đo và cố định cả hai con số. Nếu một thay đổi làm dịch chuyển một trong hai con số, đó là quyết định sản phẩm, không phải refactor.

Trục `TEACHER_SOURCE` cố tình vắng mặt: precomputed và GEMM-produced teacher logits có ngữ nghĩa bộ nhớ phía kernel giống hệt nhau; tách chúng sẽ nhân đôi binaries mà không thay đổi mã sinh ra.

### 3.2 Runtime scalars so với specialization identity

`kl_alpha`, `kl_beta`, `entropy_gate`, và cả hai temperature vẫn là plan scalars thay vì các giá trị specialization trực tiếp. Các giá trị không dùng được canonicalize (`0.5`, gate `None`, temperature `1.0` khi KL không hoạt động) để các plan tương đương so sánh bằng nhau. Chỉ các predicate bị chặn dẫn xuất từ các scalar đó mới vào kernel identity: `BOTH_T_ONE` cho temperature scaling và `RENYI_NEAR_ONE` cho Rényi path ổn định về số học. Đưa một scalar liên tục vào specialization sẽ bùng nổ số lượng compile và autotune identities.

### 3.3 Pointer layout

- `TIED_LAYOUT=True`: `S_ptr` chứa student rows tiếp theo là teacher rows; kernel địa chỉ hóa teacher row `i` thành `i + n_rows`; `T_ptr` bị bỏ qua.
- `TIED_LAYOUT=False`: `S_ptr` và `T_ptr` là các buffer logit row-major độc lập (dùng chung bởi separate và precomputed mode).
- Khi một constexpr path không thể đọc hay ghi teacher pointer, wrapper có thể alias nó vào một buffer đang hoạt động; nhánh bị tắt được loại bỏ lúc compile. Không bao giờ truyền pointer null/thiếu kích thước.
- `KERNEL_STAGE=ALIGNED_STATS` tái sử dụng `L_ptr` cho NLL output và `T_L_ptr` cho student LSE đã lưu. `ALIGNED_GRAD` đọc LSE từ `T_L_ptr` và hệ số per-row từ `K_ptr`. Không pointer riêng cho stage nào được mở rộng chữ ký chuẩn.

### 3.4 Bất biến an toàn label

Strict mode kiểm tra phạm vi label trên host; `assume_valid_labels=True` chỉ loại bỏ đồng bộ hóa CUDA-to-host đó. Kernel phải luôn tính

```python
y_safe = tl.where(ignored | (~in_range), 0, y)
```

và dùng `y_safe` cho mọi load và so sánh dựa trên label. `ignore_index` (ví dụ `-100`) là input API hợp lệ nhưng không bao giờ được chạm tới memory indexing. Không thêm code path nào index bằng label thô, kể cả sau trusted flag.

### 3.5 Contract biên số học

Các hành vi này là một phần của contract public và được cố định bởi differential tests; kernel và Torch reference phải dịch chuyển cùng nhau:

- Phép nhân bảo toàn zero cho các term `p·(log p − log q)`: `p == 0` đóng góp chính xác `0` cả về giá trị và gradient.
- Rows hướng ngược an toàn với support: `p > 0` nơi `q == 0` cho ra row `+inf` với gradient row bằng không.
- Dead precomputed teacher rows (toàn `-inf`): teacher CE và KL ghim về `0`; row vẫn tính trong mẫu số `mean`.
- `-inf` là support sentinel chỉ cho raw cached logits. Non-finite logits sinh ra từ GEMM tied/separate/cached-hidden xuất hiện dưới dạng `NaN` loss và gradients hoạt động thay vì bị phân loại lại thành dead rows.
- Bất ổn support của Rényi/ABKD cho ra `+inf` với gradient row KL bằng không; teacher numerics không hợp lệ theo contract `NaN` nhận biết nguồn ở trên.
- Fused kernel ghi zero chính xác vào các row logits-gradient bị ignore. Các GEMM weight-gradient nội bộ tin cậy contract đó và chỉ mask hidden rows, điều vẫn cần thiết vì số học IEEE làm `0 * NaN` / `0 * Inf` thành non-finite. Helper tổng quát vẫn mask cả hai toán hạng trừ khi trusted path được chọn tường minh.
- DKD: TCKD hoạt động tại cả hai điểm mút teacher-target; NCKD bằng không chỉ khi teacher không có non-target support; thiếu student support → row KL `+inf`, gradient KL bằng không, CE không đổi.

## 4. Autograd ownership ABI

Tám lớp `torch.autograd.Function` giữ chữ ký và gradient slot cố định:

| Function | Inputs sau `ctx` | Quyền sở hữu teacher gradient |
|---|---:|---|
| `TiedFrozenCEFunction` | 7 | không |
| `TiedTrainedCEFunction` | 7 | teacher CE qua shared weight/hidden |
| `TiedCotrainedCEFunction` | 7 | teacher KL qua shared weight/hidden, tùy chọn teacher CE |
| `PrecomputedCEFunction` | 9 | không |
| `PrecomputedAlignedSpanFunction` | 8 | không; chỉ student hidden và weight |
| `SeparateFrozenCEFunction` | 8 | không |
| `SeparateTrainedCEFunction` | 8 | teacher CE |
| `SeparateCotrainedCEFunction` | 8 | teacher KL, tùy chọn teacher CE |

Quy tắc:

- Các nhánh objective không hoạt động trả về `None` cho các input ngoài graph.
- Các buffer gradient đã cache được scale tại chỗ trong backward; chuyển đổi dtype chỉ cấp phát khi output dtype khác. Điều này tiêu thụ các buffer — một backward cho mỗi forward là contract, không phải lỗi.
- Không hợp nhất các Function qua các mode, và không để một helper dưới một Function chạm vào slice của mode khác. Chỉ chia sẻ helper dưới ranh giới Function.

## 5. Thiết kế aligned-span pipeline

Span objective không thể hoàn thành trong một lần quét vocabulary: hệ số span phụ thuộc vào mọi token NLL trong span, và gradient logits của mỗi token cần hệ số đó. Do đó có hai kernel stage quanh một phép rút gọn FP64 phía host:

1. GEMM student theo chunk + `ALIGNED_STATS` phát ra token NLL FP32 và row LSE.
2. Toán FP64 prefix/difference-array rút gọn các span và tạo một hệ số vô hướng cho mỗi token row.
3. `ALIGNED_GRAD` chuyển LSE đã lưu + hệ số thành `a_i * (p − onehot)` ghi lại lên logits buffer.

Các ràng buộc thiết kế cần giữ:

- Các lần chạy một chunk tái sử dụng logits buffer của stage 1; các lần chạy nhiều chunk tính lại chunk-local logits để bộ nhớ vocabulary đỉnh giữ ở mức chunk-local.
- Các span không có token student hợp lệ bị loại khỏi loss và mẫu số mean; precomputed builder từ chối các teacher span không có bằng chứng.
- FP64 workspace là `O(BT + spans)` và không giảm khi chia chunk; OOM của nó phải phân biệt được với chunk-vocabulary OOM có thể retry.
- `binary_kl` và `binary_ce` chia sẻ fused stages và logits gradient, do đó phải tiếp tục chia sẻ một autotune winner.

## 6. Chunking và khôi phục OOM

- `ChunkPolicy` (phân hoạch ban đầu, trần retry) được giải quyết lúc planning; dispatcher sở hữu vòng lặp retry: khi gặp CUDA/HIP OOM có thể retry, nó giải phóng allocator blocks, nhân đôi số chunk, retry đến `max_chunks_limit`, và cache phân hoạch thành công trong một bounded LRU khóa bởi tensor signatures cộng specialization identity.
- Các low-level op định tuyến qua cùng engine — không bao giờ cấp cho một direct API một vòng lặp thực thi riêng.
- Chỉ các allocation failure khớp OOM classifier mới retry được; mọi thứ khác truyền lên trên.

## 7. Autotune identity

Autotune benchmark full-batch pipelines, không phải các kernel call riêng lẻ. Một `FullBatchLaunchRequest` đã đóng băng mang toàn bộ launch và workload identity vào một selection call cho mỗi autograd Function.

Persistent store (SQLite, WAL, schema version 1) khóa winners theo:

- device/toolchain fingerprint;
- workload và pipeline keys;
- một AST digest của mọi timed source module, chuẩn hóa để ổn định qua các phiên bản CPython được hỗ trợ (3.10–3.13: docstrings bị lược, các trường AST thay đổi theo phiên bản như `type_params` bị loại);
- generation fencing và cross-process device leases có hết hạn.

Hệ quả cho các thay đổi:

- Chạm vào kernel chuẩn, một mode pipeline, các runtime math helper, dispatcher, hoặc Torch benchmark reference sẽ thay đổi timed-contract digest và buộc một autotune identity mới. Đó là chủ đích — các winner cũ không được sống sót qua một thay đổi số học.
- Các wheel không có source dùng một digest được checked-in bảo vệ bởi CI: tạo lại fallback digest bất cứ khi nào timed surface thay đổi.
- Một tiến trình benchmark trên một thiết bị vật lý tại một thời điểm (chờ lease có giới hạn); contention fallback về winner gần nhất hoặc default config thay vì làm tắc nghẽn huấn luyện. Schema recovery xác minh lại database dưới một per-store lock trước khi thay thế các file sở hữu.

## 8. Checklist mở rộng

### 8.1 Mọi mở rộng

1. Thêm product capability trước tiên trong `_planning/capabilities.py`; các kết hợp không được hỗ trợ phải bị từ chối lúc planning, không phải lúc launch.
2. Cập nhật kernel chuẩn và Torch reference cùng nhau.
3. Giữ chữ ký autograd theo mode cố định; chia sẻ helper dưới Function.
4. Thêm CPU oracle tests, CUDA differential tests, ABI tests, specialization-count tests, và autotune contract coverage.
5. Tạo lại fallback timed-contract digest nếu timed surface thay đổi.

### 8.2 Thêm một biến thể KL

1. Chỉ thêm một style có loss và cả hai hướng gradient đều xác định (student luôn luôn; hướng teacher ít nhất phải được chỉ định là detached).
2. Mở rộng `VALID_KL_VARIANTS`, literal `KLVariant`, và — nếu nó nhận hyperparameters — `KL_HYPERPARAM_SPECS` với các biên đóng/mở chính xác và default chuẩn.
3. Gán mã `KL_STYLE` tiếp theo; hiện thực nhánh Triton trong kernel chuẩn và row helper trong `_torch_backend.py`, bao gồm chính sách zero-support và sentinel (mục 3.5).
4. Quyết định detached hay teacher-KL-gradient. Một biến thể teacher-gradient gia nhập `TEACHER_KL_GRAD_VARIANTS`, bị giới hạn ở các cotrained path tied/separate, và phải từ chối temperature bất đối xứng.
5. Cập nhật các số specialization đã cố định và các bảng tài liệu ([variants.md §4](variants.md#4-các-biến-thể-kl-divergence), [README §KL variants](README.md#kl-variants)).

### 8.3 Thêm một teacher mode / pipeline

1. Vocabulary mới trong `_planning/models.py`, capability entry, routing trong `resolve_kernel_path`, gradient policy trong `teacher_grad_policy`.
2. Một vertical slice mới dưới `ops/_kernels/<mode>/` với autograd `Function` riêng (ABI cố định) và benchmark builder; tái sử dụng kernel chuẩn — không thêm một thân `@triton.jit` thứ hai.
3. Một module low-level op public chỉ export hàm `*_distillation_loss`.
4. Nối dispatcher executor resolution và chunk-cache key nếu tập tensor khác biệt.

## 9. Test gates

| Gate | Cố định |
|---|---|
| `test_kernel_specializations.py` | 170 legacy identity / 165 canonical semantic combination counts. |
| CPU oracle suites | Torch reference so với fp64 references, mọi variant và edge sentinels. |
| `test_public_api_cuda.py`, `test_low_level_ops_cuda.py` | Differential Triton vs Torch: mọi teacher path, họ KL, các nhánh không hoạt động, ignored non-finite rows (strict + trusted), noncontiguous inputs, chuyển đổi dtype precomputed, aligned-span stages. |
| `test_dlogits_contract_cuda.py` | Contract zero chính xác cho ignored-row logits-gradient được tiêu thụ bởi hidden-only weight-gradient masking path, trên mọi teacher mode và aligned-span execution. |
| `test_cartesian_cuda.py` (sharded) | Ma trận public đầy đủ: weights, label failures, layouts, chunk policies, APIs, dtypes. |
| `test_autotune_full_batch_cuda.py` | Bounded full-plan timing contract, launch 16K/32K, coverage `autotune="Max"`. |
| Multiprocess cache gate (thủ công) | Cold WAL/schema locking với 8 spawn workers, lease ownership, OOM-retry double xác định. |

Xem [gpu-validation.md](gpu-validation.md) cho ma trận lệnh và các workflow của repository cho phạm vi CI tự động.
