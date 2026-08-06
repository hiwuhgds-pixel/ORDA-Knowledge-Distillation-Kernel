# Phạm vi CI

Các kiểm tra tự động trong repository này chạy CPU-only, nên pull requests và releases không phụ thuộc vào GPU runner vận hành riêng.

Kiểm tra pull-request bao gồm formatting và linting, type checking, autotune contract digest, xây dựng và xác minh wheel, và bộ test CPU trên các endpoint Python/Torch được hỗ trợ. Kiểm tra release build một wheel, test artifact đã cài trên CPU, sau đó publish khi các kiểm tra đó đạt.

Các test GPU correctness, Cartesian, autotune, performance, và VRAM nằm trong source tree để dùng thủ công trên phần cứng có CUDA hoặc ROCm Torch runtime tương thích và Triton. GitHub Actions không chạy chúng; theo [gpu-validation.md](gpu-validation.md) khi có phần cứng như vậy.

Không có scheduled nightly workflow hay automated benchmark gate. Các lệnh benchmark dành cho việc chạy có chủ đích tại local hoặc dedicated runner, nơi các phép đo thời gian và bộ nhớ có thể diễn giải trong bối cảnh của máy tạo ra chúng.
