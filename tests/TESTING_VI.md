# Bộ thử nghiệm và đo hiệu năng (Test & Benchmark Suite) của orda_ce_kernel

Bộ test này được thiết kế để kiểm thử gói đã cài đặt (installed-package testing) theo bố cục cấu trúc thư mục `src/`.
Từ một môi trường mới, hãy cài đặt gói trước:

```
python -m pip install -e ".[test]"
```

## Phần cứng mục tiêu (Target Hardware)

Orda được viết để dùng cho cho **Tesla T4 / fp16**.

**Không áp dụng các số liệu benchmark của Orda sang các dòng chip A100/H100** — các quyết định tối ưu hóa kiến trúc ở đây hiện chỉ được kiểm thử trên Tesla T4. Hãy chạy lại kiểm thử trên phần cứng mục tiêu trước khi đưa ra các tuyên bố về hiệu năng.

Các kiểm thử độ chính xác số học (CUDA/Triton correctness tests) và tất cả các script benchmark yêu cầu môi trường Linux CUDA/HIP thực tế nơi `orda_ce_kernel.is_available()` trả về `True`. Nếu điều kiện này không được đáp ứng, các kiểm thử GPU sẽ tự động bỏ qua (skip) và các script benchmark sẽ kết thúc sớm mà không đưa ra số liệu.

## Trình chạy một click (Colab / T4)

```
python scripts/run_all_test_colab.py
```

Trình chạy tự động một click này sẽ lưu các kết quả (artifacts) vào thư mục chỉ định qua cờ `--output-dir` (mặc định là `benchmark_results/`) theo bố cục cấu trúc sau:

```
benchmark_results/
    json/    # một file JSON cho mỗi bước benchmark
    csv/     # một file CSV cho mỗi bước benchmark
    logs/    # một file .log cho từng bước + file log chính run_all.log
```

Các cờ hữu ích:

- `--skip-large` — bỏ qua cấu hình `bench_end_to_end --mode orda-large` (16x1024) khi dung lượng VRAM trống bị hạn chế.
- `--skip-correctness` / `--skip-unit` — bỏ qua các cổng kiểm thử pytest.
- `--dry-run` — chỉ in danh sách các lệnh dự kiến chạy mà không thực thi chúng.

## Bố cục cấu trúc (Layout)

- `tests/unit/`: Các kiểm thử an toàn trên CPU để xác thực public API, teacher objects, `DistillationLoss`, PyTorch fallback, validation trước CUDA dispatch, hành vi resolver/dispatcher, logic phân phối lại khi gặp OOM, các hàm trợ giúp quant/dequant, thống kê timing, artifact benchmark, và luồng runtime option.
- `tests/correctness/`: Các kiểm thử số học trên CUDA/Triton so với các hàm tham chiếu PyTorch/FP64. Các kiểm thử này bao gồm CE-only, CE+KL, KL độc lập, tied/separate/precomputed KD modes, default KD weights, raw loss component contract, upstream autograd scaling, các chế độ reduction được API hỗ trợ, kích thước vocab không phải lũy thừa của hai, phạm vi hỗ trợ dtype, phân chia chunk, `max_fused_size`, label smoothing, `ignore_index`, các logit cực hạn, gradient hữu hạn, và deterministic output.
- `tests/benchmarks/`: Các điểm chạy benchmark cho CE-only, CE+KL, độ chính xác KL, throughput KL, ước lượng băng thông bộ nhớ, các luồng huấn luyện giả lập end-to-end, và một wrapper Nsight để profile.
- `tests/utils/`: Các hàm kiểm soát runtime dùng chung, tham chiếu số học FP64, các thang đo so sánh, đo thời gian bằng CUDA event, xử lý lỗi OOM, và các hàm hiển thị kết quả đầu ra.
- `tests/VALIDATION_EN.md` / `tests/VALIDATION_VI.md`: Bộ tài liệu hướng dẫn về cổng kiểm thử và phê duyệt số liệu.
- `scripts/check_ast_imports.py`: Cổng kiểm tra AST và tính nguyên bản của import đa nền tảng.
- `scripts/run_all_test_colab.py`: Script chạy tổng hợp đo đạc trên T4/Colab.

## Cài đặt và Kiểm tra Tĩnh (Static Checks)

```
python -m pip install -e ".[test]"
python -m pytest tests/unit -q
python scripts/check_ast_imports.py
```

## Độ chính xác CUDA (CUDA Correctness)

Chỉ chạy trên máy có hỗ trợ GPU:

```
python -m pytest tests/correctness -q
```

Các phạm vi được bao phủ kiểm thử:

- Loss và gradient của CE-only và CE+KL khớp với các tham chiếu PyTorch/FP64.
- `student_ce`, `teacher_ce`, `kl` là các component được report; tổng loss phải dùng đúng `student_ce_weight`, `teacher_ce_weight`, và `kd_weight`.
- Standalone KL kernel khớp với PyTorch trên nhiều mức nhiệt độ (temperatures), online/fixed softmax, và các tổ hợp cờ fast-math khác nhau.
- `teacher_ce_weight=None` mặc định là pure KD (`0.0`) cho `SeparateTeacher` và `PrecomputedTeacher`, nhưng vẫn là teacher CE (`1.0`) cho `TiedTeacher`.
- `DistillationLoss` phải tương đương functional API và backward phải tôn trọng upstream gradient scale.
- Gradient ẩn (hidden-state gradients) đối với các token bị bỏ qua (ignore index) phải bằng không.
- Kiểm tra các kiểu reduction được hỗ trợ là `mean` và `sum`.
- Các kích thước đầu vào bao gồm batch nhỏ, số lượng token lớn, và kích thước từ vựng (vocab) không phải lũy thừa của hai.
- Các kiểu dữ liệu (dtype) bao gồm fp16, fp32, và bf16 khi GPU thông báo hỗ trợ bf16.
- Các trường hợp logit cực hạn vẫn phải giữ cho loss và gradient ở mức hữu hạn (không bị NaN/Inf).

## Đo hiệu năng (Benchmarks)

Các benchmark không chứa script fallback chạy trên CPU. Nếu CUDA/Triton không khả dụng, chúng sẽ in thông báo bỏ qua và thoát mà không tạo ra số liệu.

```
python -m tests.benchmarks.bench_ce_only
python -m tests.benchmarks.bench_ce_kl
python -m tests.benchmarks.bench_kl_accuracy
python -m tests.benchmarks.bench_kl_throughput
python -m tests.benchmarks.bench_memory_bandwidth --verify
python -m tests.benchmarks.bench_end_to_end
```

Ví dụ chạy nhanh (Smoke examples):

```
python -m tests.benchmarks.bench_ce_only --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2
python -m tests.benchmarks.bench_ce_kl --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2
python -m tests.benchmarks.bench_memory_bandwidth --configs 1x32 --vocab-size 1024 --hidden-dim 128 --warmup 1 --steps 2 --verify
```

### Chi tiết các tham số CLI của từng Script

Ngoài các tham số chung của CLI (`--configs`, `--vocab-size`, `--hidden-dim`, `--dtype`, `--warmup`, `--steps`, `--repeats`, `--seed`), một số script benchmark chấp nhận các tham số riêng hoặc ghi đè giá trị mặc định:

1. **`bench_kl_accuracy.py`**:
   * `--sample-frac` (mặc định: `0.25`): Tỷ lệ lấy mẫu bộ dữ liệu kiểm chứng để đánh giá.
   * *Ghi đè*: `--hidden-dim` mặc định là `1024`.
2. **`bench_kl_throughput.py`**:
   * `--student-layers` (mặc định: `4`): Số lớp của mô hình student được mô phỏng.
   * `--teacher-layers` (mặc định: `12`): Số lớp của mô hình teacher được mô phỏng.
   * `--grad-accum` (mặc định: `4`): Số bước tích lũy gradient (gradient accumulation).
   * `--lambda-student` (mặc định: `1.0`): Hệ số nhân tính toán cho loss của student.
   * `--no-compile` (cờ): Vô hiệu hóa biên dịch `torch.compile` trên mô hình.
   * *Ghi đè*: `--hidden-dim` mặc định là `1024`; `--warmup` mặc định là `2`.
3. **`bench_end_to_end.py`**:
   * `--layers` (mặc định: `2`): Số lớp của Transformer được mô phỏng.
   * `--heads` (mặc định: `8`): Số attention heads được mô phỏng.
   * `--batch-size` (mặc định: `None`): Ghi đè kích thước batch của chế độ chạy.
   * `--seq-len` (mặc định: `None`): Ghi đè độ dài chuỗi của chế độ chạy.
   * `--no-compile` (cờ): Vô hiệu hóa việc biên dịch mô hình.
   * *Ghi đè*: `--hidden-dim` mặc định là `2048`; `--steps` mặc định là `15`.
4. **`profile_wrapper.py`**:
   * `--target-module` (mặc định: `"tests.benchmarks.bench_ce_only"`): Mô-đun benchmark mục tiêu cần profile.
   * `--target-args` (mặc định: `""`): Các đối số bổ sung truyền cho script mục tiêu.
   * `--ncu` (mặc định: `None`): Đường dẫn tuyệt đối đến chương trình Nvidia NCU.
   * `--nsys` (mặc định: `None`): Đường dẫn tuyệt đối đến chương trình Nvidia Nsys.
   * `--kernel` (mặc định: `None`): Bộ lọc theo tên của kernel cần phân tích.
   * *Ghi đè*: `--output-dir` mặc định là `profile_results`.

## Kết quả Benchmark đầu ra

Mỗi script benchmark chấp nhận:

```
--output-json benchmark_results/name.json
--output-csv benchmark_results/name.csv
```

Khi các benchmark được chạy trực tiếp, các file JSON/CSV sẽ được ghi chính xác vào các đường dẫn được truyền qua CLI. Các lượt chạy trực tiếp này sẽ không tạo thư mục `logs/` trừ khi người gọi chuyển hướng stdout/stderr riêng.

File JSON chứa `metadata` (tên benchmark, mốc thời gian, phiên bản PyTorch/CUDA, thông tin thiết bị, các tham số CLI) và các dòng `rows` (mỗi dòng đại diện cho một config/method đi kèm với latency, peak VRAM, và trạng thái status).

### Các cột thống kê thời gian chạy (Timing statistics columns)

Mỗi đo đạc hiệu năng tốc độ chạy qua `--repeats` vòng lặp ngoài × `--steps` lượt đo:
- Các tham số mặc định: `--repeats 5 --steps 20` (tổng số 100 mẫu đo), ngoại trừ `bench_end_to_end` sử dụng `--steps 15` (tổng cộng 75 mẫu đo).
- Số bước khởi động mặc định: `--warmup 5`, ngoại trừ `bench_kl_throughput` sử dụng `--warmup 2`.

Các trường thông số:
- `latency_ms` — trung bình đã cắt tỉa (trimmed mean - 10%) của tất cả các mẫu đo.
- `latency_ms_std`, `latency_ms_min`, `latency_ms_max` — độ phân tán thô tối thiểu, tối đa và độ lệch chuẩn.
- `latency_ms_p50`, `latency_ms_p95` — độ trễ trung vị và phân vị đuôi 95% đáng tin cậy.
- `cv_pct` — hệ số biến thiên (coefficient of variation). Nếu `cv_pct > 15%`, script sẽ in ra cảnh báo `[WARN]` — không nên sử dụng dòng dữ liệu đó cho các tuyên bố báo cáo chính thức.

## Phân tích hiệu năng (Profiling)

```
python -m tests.benchmarks.profile_wrapper --dry-run
python -m tests.benchmarks.profile_wrapper --mode all --target-module tests.benchmarks.bench_ce_kl
```

## Các trường hợp bỏ qua hợp lệ (Valid Skips)

Bỏ qua kiểm thử được coi là hợp lệ khi:

- Không có sẵn CUDA/HIP.
- Không thể import được Triton kernels hoặc `orda_ce_kernel.is_available()` là false.
- Yêu cầu bf16 trên một GPU không hỗ trợ kiểu dữ liệu này. Các kiểm thử correctness sẽ tự động bỏ qua (skip) bf16 một cách êm đẹp; trong khi các script chạy benchmark sẽ ném ra lỗi RuntimeError nếu cố tình chạy bf16 trên phần cứng không hỗ trợ.

## Checklist trước khi thu thập số liệu thực tế

1. Cài đặt môi trường bằng lệnh `python -m pip install -e ".[test]"`.
2. Chạy kiểm tra tĩnh `python scripts/check_ast_imports.py`.
3. Chạy `python -m pytest tests/unit -q`.
4. Trên GPU, chạy kiểm correctness `python -m pytest tests/correctness -q`.
5. Chạy các config smoke benchmark để kiểm tra đầu ra JSON/CSV.
6. Chỉ chạy benchmark đầy đủ sau khi cổng kiểm tra correctness đã vượt qua hoàn toàn.
