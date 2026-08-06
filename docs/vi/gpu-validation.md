# Xác minh GPU

Chạy các gate này với bản PyTorch CUDA hoặc ROCm và Triton đã cài đặt. Bắt đầu từ một tiến trình sạch để trạng thái JIT và autotune mang tính đại diện.

## Correctness cốt lõi

```bash
python -m pytest test/correctness/test_dlogits_contract_cuda.py -q
python -m pytest test/correctness/test_public_api_cuda.py -q
python -m pytest test/correctness/test_low_level_ops_cuda.py -q
```

Các suite này so sánh mọi teacher path và họ KL với các reference Torch hoặc fp64 độc lập, bao gồm inactive branches, ignored non-finite rows dưới strict và trusted-label modes, noncontiguous inputs, và chuyển đổi dtype của precomputed logits. `test_public_api_cuda.py` cũng so sánh aligned-span Triton path với torch reference cho cả hai binary objectives, CE bật/tắt, và một so với nhiều chunks; nó kiểm tra loss, hidden gradients, và weight gradients. `test_dlogits_contract_cuda.py` cố định contract zero cho ignored-row được dùng bởi trusted weight-gradient accumulation path trên mọi teacher mode và aligned-span kernel.

Các case aligned-span là hardware gate cho cả hai canonical stages:

- `ALIGNED_STATS`: NLL an toàn với ignored-row và LSE đã lưu
- `ALIGNED_GRAD`: `a_i * (p - onehot)` ghi lại vào logits buffer
- tái sử dụng logits một chunk và tính lại nhiều chunk
- toán FP64 span prefix/difference quanh thống kê FP32 của kernel

## Support-sentinel và block-independence gates

```bash
python -m pytest test/correctness/test_kl_variants_cuda.py -q
```

Ngoài parity giữa kernel và reference, suite này cố định các trường hợp support nhạy cảm về số học:

- Rényi và Rényi-reverse trên các cột teacher có `-inf` support, đối chiếu với fp64 reference tại partial base support
- `reverse`, `jeffreys`, và `entropy_adaptive` trả về loss `+inf` và gradients bằng không khi thiếu support qua `max_fused_size` trong `{32, 128, 512}`, để quyết định sentinel không phụ thuộc vào per-block normalization đang chạy
- `alpha_divergence` tại `alpha` gần `0` và `1` khớp giá trị reference và giữ không âm
- frozen tied và separate teachers để gradients `teacher_hidden`/`teacher_weight` là `None` trong khi student path vẫn nhận gradient

## Ma trận Cartesian đầy đủ

GPU runner sở hữu các file correctness trực tiếp, full-batch autotune gate, và Cartesian. Chọn một hoặc nhiều suite chính xác với `--test` lặp lại được:

```bash
python test/run_test_gpu.py --list
python test/run_test_gpu.py --test public-api
python test/run_test_gpu.py --test low-level --test kl-variants
python test/run_test_gpu.py --test autotune
python test/run_test_gpu.py --all
```

Suite Cartesian bao gồm `student_ce_weight=0`, strict label failures, trusted layouts, mọi chunk policy, mọi API public/direct, và các dtype được hỗ trợ. Các case thực thi dưới dạng pytest items nhóm theo `(teacher_representation, kl_variant, dtype, api_level)`. Runner có bốn teacher-representation routes; KL variant là bộ lọc độc lập. Không có selection flags, nó in cùng kiểu chọn tương tác đánh số như các benchmark router. Bốn Cartesian routes là các option trực tiếp 6-9:

```bash
python test/run_test_gpu.py
python test/run_test_gpu.py --test cartesian-tied
python test/run_test_gpu.py --route tied --kl forward --dtype float32
```

Chọn option 6-9 chạy cả bốn route trong các pytest subprocess tuần tự riêng biệt thay vì tạo ma trận 2.073.600 case trong một tiến trình. Cartesian luôn chạy với autotune tắt; các suite autotune chuyên biệt sở hữu phần coverage đó. Bộ lọc cũng bao gồm `--kl` lặp lại được, cùng `--api`, `--chunk`, `--labels`, `--outcome`, và `--ordinal` lặp lại được.

Chia shard bất kỳ route đã chọn qua nhiều worker hoặc máy:

```bash
export ORDA_CARTESIAN_SHARD_COUNT="8"
export ORDA_CARTESIAN_SHARD_INDEX="0"  # 0..7
python test/run_test_gpu.py --route tied --kl forward -q
```

Chạy mọi shard index với cùng shard count.

Chạy lại các case cụ thể theo ordinal, không cần biến shard:

```bash
python test/run_test_gpu.py --ordinal 123456 --ordinal 123457 -q
```

## Autotune và block lớn

```bash
python test/run_test_gpu.py --test autotune -q
```

Gate này chạy ma trận supported-path, xác minh bounded full-plan timing contract, so sánh các launch 16K/32K rõ ràng với Torch, và khẳng định rằng budget 32K bộc lộ cả hai block size lớn. `autotune=True` và `"Max"` dùng cùng candidate set; `"Max"` thực hiện phép đo robust sâu hơn.

Các suite GPU không được lên lịch trong CI của repository này. Chạy các lệnh liên quan thủ công trên từng kiến trúc CUDA hoặc ROCm cần hỗ trợ, bao gồm ROCm launch-candidate coverage khi áp dụng. Ghi lại chi tiết phần cứng và runtime cùng kết quả.

## Multi-device guard

Trên máy có ít nhất hai GPU nhìn thấy được:

```bash
python test/run_test_gpu.py --test public-api -- \
  -k "current_device_differs" -q
```

Test giữ ambient device tại `cuda:0`, thực thi tensors trên `cuda:1`, và kiểm tra launch guard khôi phục ambient device.

## GPU gate cuối

```bash
python test/run_test_gpu.py --all -q
```

Gate cuối chạy mỗi direct suite và autotune suite một lần, sau đó thực thi Cartesian dưới dạng bốn teacher-representation route tuần tự. Ghi lại GPU model, phiên bản Torch, phân phối/phiên bản Triton, CUDA hoặc HIP runtime, và phiên bản driver cùng kết quả.
