# Các cổng xác thực (Validation Gates)

Hãy chạy kiểm thử qua các cổng này trước khi sử dụng các số liệu đo hiệu năng (benchmark) trong báo cáo chính thức.

## Gate 0: Phạm vi và Tính nguyên bản của Import (Scope & Import Hygiene)

Chạy từ thư mục gốc của repository trên bất kỳ máy nào. Script kiểm tra viết bằng Python thuần túy nên chạy được trên cả Linux, macOS và Windows:

```
python scripts/check_ast_imports.py
```

Kết quả mong đợi:

- Quá trình phân tích cú pháp AST thành công cho mọi file Python trong thư mục `tests/`.
- Không phát hiện việc sử dụng `sys.path` / `sitecustomize` / `PYTHONPATH` để thay đổi đường dẫn import trong mã nguồn test.
- Không có file nào trong `src/**` bị sửa đổi để phục vụ kiểm thử.

## Gate 1: Kiểm thử Unit an toàn trên CPU (CPU-Safe Unit Tests)

```
python -m pytest tests/unit -q
```

Kết quả mong đợi: tất cả các bài unit test đều vượt qua. Gate này dùng để xác thực:

- Các hàm xuất khẩu public API, `DistillationLoss`, teacher objects, và các lỗi xác thực tham số đầu vào trước CUDA dispatch;
- Luồng runtime option qua public API arguments và `KernelConfig`;
- Core runtime modules nhận execution options qua function arguments;
- Kích thước chia chunk của resolver và các giới hạn chunk tối đa;
- Hành vi thử lại khi OOM và cơ chế lưu bộ đệm (cache) của dynamic dispatcher thông qua giả lập (mocks);
- Các hành vi liên quan đến shape của quant/dequant, các dòng toàn số 0, seed stochastic ngẫu nhiên và các dòng mục tiêu (target rows);
- Các phép tính toán học thống kê đo thời gian chạy (trimmed mean, stdev, percentile, logic hệ số biến thiên cv_pct);
- Phân tích cú pháp của các công cụ benchmark, ghi xuất các file báo cáo (artifacts), cơ chế fallback eager được bảo vệ khi biên dịch thất bại và các alias kiểu dữ liệu (dtype).

## Gate 2: Độ chính xác CUDA (CUDA Correctness)

Chỉ chạy trên máy Linux có hỗ trợ CUDA/HIP nơi `orda_ce_kernel.is_available()` là `True`:

```
python -m pytest tests/correctness -q
```

Kết quả mong đợi: tất cả các bài kiểm thử độ chính xác trên GPU đều vượt qua. Các thành phần bao phủ gồm:

- Fused loss CE+KL và các gradient của chúng so với tham chiếu FP64;
- Các phép reduction `mean` và `sum` cho riêng nhánh CE;
- Cơ chế chia chunk tường minh (explicit) và động (dynamic);
- Cơ chế che giấu gradient (masking) đối với các token bị bỏ qua thông qua `ignore_index`;
- Các tham số label smoothing và `student_ce_weight`;
- Contract `student_ce`, `teacher_ce`, `kl` là reported component, và total loss áp dụng đúng `student_ce_weight`, `teacher_ce_weight`, `kd_weight`;
- Default `teacher_ce_weight=None` trong `SeparateTeacher`/`PrecomputedTeacher` là pure KD, còn `TiedTeacher` giữ teacher CE;
- `DistillationLoss` khớp functional API và backward tôn trọng upstream gradient scale;
- Các kích thước từ vựng (vocabulary size) không phải lũy thừa của hai;
- Các đầu vào kiểu dữ liệu fp16, fp32 và bf16 (nếu được phần cứng hỗ trợ);
- Standalone Triton KL loss/gradient;
- Các cờ cấu hình liên quan đến KL temperature, online softmax, fast-math, và multiply-not-divide;
- Các dòng dữ liệu bị bỏ qua hoàn toàn trong phép tính KL;
- Stress test số học dưới các điều kiện logit cực hạn;
- Xác thực gradient của mô hình có cấu hình chia sẻ trọng số (tied weight - embedding và head dùng chung);
- Đầu ra giống hệt nhau khi nhận các đầu vào giống nhau (deterministic).

## Gate 3: Thử nghiệm Benchmark nhanh (Benchmark Smoke)

Chỉ chạy sau khi Gate 2 đã vượt qua:

```
python -m tests.benchmarks.bench_ce_only --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2 --output-json benchmark_results/ce_only_smoke.json --output-csv benchmark_results/ce_only_smoke.csv
python -m tests.benchmarks.bench_ce_kl --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2 --output-json benchmark_results/ce_kl_smoke.json --output-csv benchmark_results/ce_kl_smoke.csv
python -m tests.benchmarks.bench_memory_bandwidth --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2 --verify --output-json benchmark_results/memory_smoke.json --output-csv benchmark_results/memory_smoke.csv
```

Kết quả mong đợi:

- Các script in ra thời gian chạy CUDA thực tế hoặc các dòng OOM tường minh;
- Các file JSON/CSV được tạo ra đúng yêu cầu, bao gồm các dòng có trạng thái `status=skipped` khi thư viện, CUDA hoặc Triton kernel không khả dụng;
- Metadata ghi lại đúng thông tin GPU chạy, các phiên bản CUDA/HIP/PyTorch, và các đối số CLI;
- Không có script nào tự sinh ra các số liệu giả bằng cách chạy fallback trên CPU.
- Lệnh `bench_memory_bandwidth --verify` thực hiện thành công các bước kiểm tra tính hữu hạn của loss và gradient đối với CE/CE+KL so với PyTorch trước khi đưa ra các ước tính về lưu lượng băng thông tương đối.

## Gate 4: Thu thập Benchmark đầy đủ (Full Benchmark Collection)

Chỉ chạy đo đạc cấu hình đầy đủ sau khi Benchmark Smoke đã vượt qua thành công:

```
python -m tests.benchmarks.bench_ce_only --output-json benchmark_results/ce_only.json --output-csv benchmark_results/ce_only.csv
python -m tests.benchmarks.bench_ce_kl --output-json benchmark_results/ce_kl.json --output-csv benchmark_results/ce_kl.csv
python -m tests.benchmarks.bench_kl_accuracy --output-json benchmark_results/kl_accuracy.json --output-csv benchmark_results/kl_accuracy.csv
python -m tests.benchmarks.bench_kl_throughput --output-json benchmark_results/kl_throughput.json --output-csv benchmark_results/kl_throughput.csv
python -m tests.benchmarks.bench_memory_bandwidth --verify --output-json benchmark_results/memory_bandwidth.json --output-csv benchmark_results/memory_bandwidth.csv
python -m tests.benchmarks.bench_end_to_end --output-json benchmark_results/end_to_end.json --output-csv benchmark_results/end_to_end.csv
```

Các quy tắc diễn giải số liệu:

- Bài test `bench_kl_accuracy` bao gồm các dòng: `fp32_full`, `fp16_full`, `fp16_f32_row_sum`, `sample_25`, và `triton`.
- Bài test `bench_kl_throughput` bao gồm các dòng: `no_kl`, `kl_triton`, `fp16_f32_row_sum`, `kl_full`, `chunked_pytorch_kl`, và `kl_sample_25`.
- Các dòng có trạng thái `status=ok` là các dòng chứa số liệu thời gian trễ hữu ích để so sánh.
- Các dòng có trạng thái `status=oom` biểu thị các quan sát thực tế về dung lượng bộ nhớ giới hạn, không phải là dữ liệu thời gian trễ.
- Không được đưa giá trị trễ `NaN` vào tính toán trung bình hiệu năng.
- Chỉ số ước lượng GiB/s chỉ mang tính tương đối để so sánh. Hãy sử dụng Nsight để có kết quả băng thông DRAM/L2 chính xác nhất.
- Các dòng đo thời gian thực tế qua `cuda_benchmark` (tức là các dòng `status=ok` và `status=oom` thực sự đã chạy) sẽ đi kèm các trường thông tin: `latency_ms_std`, `_min`, `_max`, `_p50`, `_p95`, `cv_pct`, và `repeats`. Các dòng bỏ qua `status=skipped` và các dòng OOM thoát sớm trước khi quá trình đo thời gian bắt đầu sẽ không có các trường thống kê này — việc này nhằm tránh việc các công cụ hạ nguồn nhầm lẫn một cấu hình chưa được đo đạc với cấu hình đã đo. Các dòng có `cv_pct > 15%` sẽ bị gắn nhãn cảnh báo `[WARN]`; chúng cần được chạy lại hoặc loại bỏ khỏi các tuyên bố chính thức của báo cáo.
- Các dòng trong `bench_kl_throughput` sử dụng các trường so sánh `vs_no_kl` và `peak_vram_delta_vs_no_kl_mib`. Các kết quả cũ chứa `vs_kl_triton` hay `peak_vram_delta_vs_kl_triton_mib` là phiên bản cũ và cần được chạy lại để tạo dữ liệu mới trước khi báo cáo.
- Các dòng benchmark có thể chứa trường `loss_compile` để thể hiện xem baseline loss của PyTorch hoặc các mô hình student/teacher có thử nghiệm biên dịch bằng `torch.compile` với eager fallback hay không.

## Gate 5: Phân tích hiệu năng chuyên sâu (Profiling)

Để xác thực cú pháp lệnh:

```
python -m tests.benchmarks.profile_wrapper --dry-run
```

Để thực thi phân tích chuyên sâu thực tế:

```
python -m tests.benchmarks.profile_wrapper --mode all --target-module tests.benchmarks.bench_ce_kl
```

Kết quả mong đợi: Các câu lệnh Nsight tương ứng được in ra trong chế độ dry-run, hoặc các tệp kết quả `.ncu-rep` / `.nsys-rep` được tạo ra khi hệ thống có sẵn các công cụ profiling và CUDA.

## Phê duyệt cuối cùng (Final Sign-Off)

Trước khi công bố hoặc báo cáo số liệu, hãy ghi chép lại:

- danh sách chính xác các câu lệnh đã thực thi;
- kết quả vượt qua của unit test và correctness test;
- dòng card GPU và các metadata phần mềm lấy từ file JSON kết quả;
- các tham số cấu hình của benchmark, kiểu dữ liệu dtype, số bước warmup, steps và seed sử dụng;
- các dòng báo lỗi OOM hoặc các bài benchmark bị bỏ qua;
- ghi rõ việc có thực hiện profiling bằng Nsight hay chỉ chạy sinh câu lệnh dry-run;
- lưu ý quan trọng về phần cứng (được kiểm thử trên T4 fp16; các con số hiệu năng không tự động đại diện cho các card Hopper/Ada).
