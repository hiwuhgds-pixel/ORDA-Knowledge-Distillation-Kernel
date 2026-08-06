# CI scope

Automated checks in this repository run CPU-only, so pull requests and releases do not depend on a privately operated GPU runner.

Pull-request checks cover formatting and linting, type checking, the autotune contract digest, wheel construction and verification, and the CPU test suite on the supported Python/Torch endpoints. Release checks build one wheel, test that installed artifact on CPU, then publish it after those checks pass.

GPU correctness, Cartesian, autotune, performance, and VRAM tests live in the source tree for manual use on hardware with a compatible CUDA or ROCm Torch runtime and Triton. GitHub Actions does not run them; follow [gpu-validation.md](gpu-validation.md) when such hardware is available.

There is no scheduled nightly workflow or automated benchmark gate. Benchmark commands are for deliberate local or dedicated-runner use, where timing and memory measurements can be interpreted in the context of the machine that produced them.
