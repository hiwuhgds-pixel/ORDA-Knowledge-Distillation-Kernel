# Kiến trúc

`orda_kernel` cung cấp các API distillation cấp token (tokenwise) và đoạn căn chỉnh (span-aligned) với hai backend Torch và Triton. Policy resolution diễn ra ở cold-path Python; vocabulary reduction và logit gradients được xử lý bởi một Triton kernel chuyên biệt hóa theo constexpr. Tài liệu này mô tả luồng thực thi runtime. Contracts cấp kernel (constexpr axes, pointer layout, autograd ABI, quy tắc mở rộng) nằm tại [designer.md](designer.md); contract dành cho người dùng nằm tại [reference.md](reference.md) và [variants.md](variants.md).

## Cấu trúc package

```text
src/orda_kernel/
├── api.py                         functional API cấp cao
├── _api_types.py                  public descriptors và cấu hình
├── _config.py                     hợp nhất profile + KernelConfig
├── _validation.py                 contracts shape/dtype/device/label
├── _torch_backend.py              reference full-logit (nguồn chuẩn số học)
├── _planning/                     policy bất biến và execution plans
│   ├── capabilities.py            ranh giới sản phẩm (các kết hợp được hỗ trợ)
│   ├── policy.py                  routing: weights + variant → kernel path
│   └── models.py                  planning vocabulary, KL hyperparameter specs
├── _execution/                    dispatch tokenwise/aligned và khôi phục OOM
├── _autotune/                     launch selection và persistent cache
└── ops/
    ├── tied_teacher.py             public low-level API cho tied
    ├── separate_teacher.py         public low-level API cho separate
    ├── precomputed_teacher.py      public low-level API cho precomputed
    ├── precomputed_inputs/
    │   ├── aligned_span.py         public API cho cross-vocabulary span
    │   ├── span_teacher.py         builder precomputed span-teacher
    │   ├── cross_vocab.py          builders chiếu token-map
    │   ├── multi_teacher.py        builders trộn multi-teacher
    │   └── _aligned_span_math.py   phép rút gọn FP64 span dùng chung
    ├── _common.py                  helpers tensor/autograd dùng chung
    └── _kernels/
        ├── ce_kl_fwdbwd.py         Triton CE/KL kernel chuẩn (kernel duy nhất được launch)
        ├── _kl_math.py             helpers toán học inline @triton.jit (không bao giờ launch)
        ├── tied/{frozen,trained,cotrained}.py
        ├── precomputed/{tokenwise,aligned_span}.py
        └── separate/{frozen,trained,cotrained}.py
```

Mỗi mode file là một vertical slice: benchmark task, vòng lặp chunk, các phép GEMM, autograd Function, và gradient ABI cố định nằm cùng nhau. Chỉ phần thân số học Triton được chia sẻ.

Ba module low-level public chỉ export các hàm `*_distillation_loss`; các lớp autograd và kernel helpers là private implementation API.

## Luồng thực thi public

1. `distillation_loss` kiểm tra tensors, hợp nhất profile/`KernelConfig`, giải quyết KL hyperparameters, và chuẩn hóa loss scalars.
2. `build_execution_plan` (hoặc `build_aligned_span_execution_plan`) giải quyết backend, kernel path, specialization, và chunk policy thành một immutable plan. Với `backend="auto"`, một policy mà không Triton specialization nào thực thi được sẽ lập lại kế hoạch sang Torch backend và phát cảnh báo fallback một lần.
3. Torch plans gọi backend reference full-logit.
4. Tokenwise Triton plans vào `execute_plan`; aligned-span plans vào executor riêng. Cả hai truyền plan và một `ChunkAttempt` tới mode slice.
5. Dynamic plans thử lại với nhiều chunks hơn sau các lỗi OOM có thể khôi phục và cache phân hoạch thành công.

## Các chế độ teacher ở runtime

**Tied:** student và teacher chia sẻ một projection weight. Logits của student và teacher được xếp chồng trong một buffer tạm.

**Tied cotrained:** mutual/cojeffreys KL dưới tied layout; teacher hidden nhận gradient KL (và CE tùy chọn); shared weight tích lũy cả hai nửa xếp chồng.

**Separate frozen:** teacher logits được tạo theo chunk, không gradient.

**Separate trained:** teacher CE nắm giữ teacher gradients; mọi biến thể KL teacher-detached đều hạ xuống Triton path.

**Separate cotrained:** mutual/cojeffreys KL nắm giữ teacher gradients, với đóng góp teacher CE tùy chọn.

**Precomputed tokenwise:** teacher logits cung cấp trực tiếp hoặc tính lại từ hidden+weight đã cache; không bao giờ có teacher gradient.

**Precomputed aligned span:** span log-likelihoods teacher đã detach điều khiển pipeline hai giai đoạn chỉ dành cho student; một `CrossVocabTeacher` live chạy trên Torch reference và có thể chuyển đổi bằng `build_precomputed_span_teacher`.

`KERNEL_CAPABILITIES` trong `_planning/capabilities.py` là ranh giới sản phẩm: kernel chuẩn có thể compile các kết hợp constexpr khác, nhưng các cặp mode/variant không được hỗ trợ bị từ chối trước khi launch. Bảng routing/eligibility đầy đủ nằm tại [variants.md](variants.md).

## Chunking và bộ nhớ

Triton backend chỉ tạo chunk-local logits. `ChunkPolicy` điều khiển phân hoạch ban đầu và giới hạn retry; một dynamic partition thành công được cache theo định danh tensor và specialization (bounded LRU). Các low-level API trực tiếp dùng cùng execution engine nên có cùng hành vi OOM retry.

Các yêu cầu chunk public dùng value object `ChunkSize` (`ChunkSize.nrows(n)` hoặc `ChunkSize.dynamic()` — kiểu request hoàn chỉnh). `num_chunks` và `max_chunks` là hai control riêng biệt có ràng buộc liên trường được kiểm tra trước khi planning.

Torch backend là reference số học và tạo toàn bộ logits `(BT, vocab)`. `backend="auto"` phát một cảnh báo thread-safe trên toàn tiến trình trước khi fallback.

## Autotune và cache

Autotune benchmark toàn bộ full-batch pipelines, không phải các kernel call riêng lẻ; một selection call diễn ra cho mỗi autograd Function. Aligned-span giữ một pipeline identity riêng; binary KL và binary CE chia sẻ một winner vì fused stages và logits gradient của chúng giống hệt nhau.

Winners nằm trong LRU in-memory cộng một SQLite WAL database được khóa bởi device fingerprint, workload/pipeline keys, timed-source AST digest, và generation fencing với cross-process device leases có hết hạn. KL hyperparameters và temperatures là runtime plan scalars — thay đổi chúng không bao giờ benchmark lại. Biến môi trường và vị trí cache được liệt kê tại [reference.md §12](reference.md#12-biến-môi-trường-autotune); contract identity/digest nằm tại [designer.md §7](designer.md#7-autotune-identity).

## Đa tiến trình

Ghi SQLite dùng transactions và WAL. Tại một thời điểm chỉ một tiến trình benchmark trên một thiết bị vật lý; thời gian chờ lease được giới hạn, và khi có contention hệ thống fallback về winner gần nhất hoặc default launch config thay vì làm tắc nghẽn huấn luyện. Schema recovery xác minh lại database dưới một per-store lock trước khi thay thế các file sở hữu.
