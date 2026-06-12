# orda_ce_kernel

Triton kernel for ORDA KL distillation + fused Cross Entropy substrate.

Documentation languages: [English](README.md) | [Tiếng Việt](README_VI.md)

Callers describe the distillation setup with teacher objects and loss weights.
Kernel options are grouped in `KernelConfig` or selected through a small
`profile` preset.

## Current Status

This repository is not yet a plug-and-play training library for most use cases.
The current version implements the core skeleton of the ORDA KL distillation
kernel and its fused Cross Entropy substrate, but validation is still limited by
the compute resources currently available to the author.

All current tests and benchmark artifacts were produced on free-tier Tesla T4
resources. Because broader GPU access is not available, the project still needs
additional correctness, stability, and performance validation across different
GPU architectures before it should be treated as a mature drop-in dependency.

This repository is being published in the hope that the community can help run
tests, benchmarks, bug reports, and experiments on more CUDA/HIP devices. That
feedback will help improve kernel portability and mature a new kernel path for
knowledge distillation, an area where open kernel implementations remain
relatively scarce.

## Install

```bash
python -m pip install -e .
```

For local validation and development checks:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/unit -q
python scripts/check_ast_imports.py
```

Triton execution currently targets CUDA-style PyTorch devices and has been
validated on CUDA/Tesla T4 fp16. HIP/ROCm support is intended through PyTorch's
CUDA/HIP device abstraction, but should be treated as experimental until
validated on real ROCm hardware. CPU calls can use the PyTorch reference
fallback with `backend="auto"` or `backend="torch"`.

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

## Demo Notebook

A sample Colab/Kaggle notebook showing how to use the ORDA kernel with HF Llama 3.2 and compare it against a torch.compile baseline.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hiwuhgds-pixel/ORDA-Knowledge-Distillation-Kernel/blob/main/notebooks/llama32_distillation_demo.ipynb)

[View the Llama 3.2 distillation demo notebook on GitHub](https://github.com/hiwuhgds-pixel/ORDA-Knowledge-Distillation-Kernel/blob/main/notebooks/llama32_distillation_demo.ipynb)

## Validation and Benchmarks

The full test and benchmark guide lives in
[`tests/TESTING_EN.md`](tests/TESTING_EN.md). CUDA/Triton correctness tests
require a real GPU environment:

```bash
python -m pytest tests/correctness -q
python scripts/run_all_test_colab.py
```

Benchmark smoke examples are documented in
[`tests/TESTING_EN.md`](tests/TESTING_EN.md#benchmarks). Current benchmark
numbers should be rerun on the target GPU before making hardware-specific
claims.

## Teacher Modes

```python
from orda_ce_kernel import TiedTeacher, SeparateTeacher, PrecomputedTeacher

teacher = TiedTeacher(hidden=teacher_hidden)
teacher = SeparateTeacher(hidden=teacher_hidden, weight=teacher_weight)
teacher = PrecomputedTeacher(logits=teacher_logits)
```

The KL Triton kernel always receives the same internal buffer layout:

```text
logits_chunk[0:n_rows]        = student logits
logits_chunk[n_rows:2*n_rows] = teacher logits
```

Teacher objects only control how that buffer is built and which gradient paths
exist. KL must read clean logits before the CE kernel overwrites the same buffer
in-place with CE gradients.

## VRAM Reduction Strategy

The main memory saving comes from avoiding full-token KL distribution tensors.
A typical PyTorch KL path materializes student log-probabilities and teacher
probabilities over the full batch-token dimension:

```text
log_p_s: [BT, V]
p_t:     [BT, V]
```

Depending on the compiled graph, an additional elementwise KL tensor with shape
`[BT, V]` may also appear.

ORDA uses the fused Cross Entropy path as a substrate for KL. Each chunk first
builds one shared logits buffer:

```text
logits_chunk[0:n_rows]        = student logits
logits_chunk[n_rows:2*n_rows] = teacher logits
```

The KL Triton kernel reads this clean buffer directly before the CE kernel
overwrites it with CE gradients. It computes the per-row KL and student KL
gradient, and only materializes the KL-specific gradient buffer:

```text
grad_kl_student: [n_rows, V]
```

That gradient is added back into `logits_chunk[:n_rows]` before the shared
buffer continues through the CE/GEMM backward lifecycle.

For example, in the T4 benchmark case
[`dim=1024 vocab=128k seq=512`](simulate/bench_vram_TiedTeacher.txt), with
`batch=16`, `BT=8192`, and `V=131072`, the PyTorch KL core tensors are:

```text
log_p_s [8192, 131072] fp16 = 2048 MiB
p_t     [8192, 131072] fp16 = 2048 MiB
total core KL tensors        = 4096 MiB
```

With ORDA dynamic chunking, `num_chunks=16` and `n_rows=512`, so the
KL-specific extra buffer is:

```text
grad_kl_student [512, 131072] fp16 = 128 MiB
```

The shared CE+KL logits buffer for that chunk is:

```text
logits_chunk [1024, 131072] fp16 = 256 MiB
```

The measured benchmark reports:

```text
torch-compile CE+KL = 8480.3 MB
orda CE+KL          = 1223.6 MB
```

Those measured values are peak extra memory for the whole CE+KL loss backend,
including CE logits, backward buffers, GEMM workspace, gradients, and allocator
overhead. The `2 * [BT, V]` to `[n_rows, V]` change above is the core KL memory
reduction, excluding any additional `[BT, V]` elementwise KL tensor that a
compiled graph may materialize.

## Performance Strategy

KL Triton reuses the `logits_chunk` already created inside the fused CE chunk
loop, so it does not need to run a separate projection/logits branch for KL. The
kernel reads clean student/teacher logits in the chunk before CE overwrites that
buffer with CE gradients, computes forward KL and the student KL gradient, then
adds the KL gradient into the student slice of the same buffer before it
continues through CE/GEMM backward.

Because KL reuses the same logits buffer and backward lifecycle as CE, adding KL
only increases latency by a small amount compared with the no-KL baseline. In
the Tesla T4 KL throughput benchmark, case
[`16x1024`](benchmark_results/logs/bench_kl_throughput.log) shows `kl_triton`
at `5778.06 ms`, compared with `5654.64 ms` for the `no_kl` baseline, or about
`2.2%` overhead.

In the experimental `TiedTeacher` training benchmark (`CE_s + CE_t + KL`), case
[`vocab=128k seq=512`](simulate/Train_TiedTeacher.txt) shows ORDA at
`1206.01 ms/step`, compared with `1357.12 ms/step` for `torch-compile`, or
about `11.1%` faster.

These numbers should be rerun on the target GPU before making hardware-specific
claims.

## Scope and Teacher Modes

This library is a logit-level forward-KL distillation kernel with a fused Cross
Entropy substrate. It is scoped around response-based KD at the output layer and
provides three teacher modes for creating or supplying teacher logits.
Feature-based KD, relation-based KD, attention transfer, reverse KL, JS
divergence, MSE-on-logits, and sequence-level KD are outside the current design
scope.

All three teacher modes build the same internal KL input layout:

```text
logits_chunk[0:n_rows]         = student logits
logits_chunk[n_rows:2*n_rows]  = teacher logits
```

The modes differ in how teacher logits are produced and which gradient paths are
available.

### `TiedTeacher(hidden)`

`TiedTeacher` is the shared-head mode. Student and teacher hidden states are
projected with the same output weight:

```text
student logits = student_hidden @ weight.T
teacher logits = teacher_hidden @ weight.T
```

In the Triton path, the student and teacher hidden chunks are concatenated and
projected with one shared GEMM. This mode is compatible with self-distillation
or online KD patterns where the teacher signal is expected to share the same
vocabulary head as the student.

By default, `teacher_ce_weight=None` resolves to `1.0` for this mode, so the
teacher CE branch is enabled unless explicitly disabled. The shared `weight` can
receive gradients from the student CE branch, teacher CE branch, and the student
side of KL. KL itself is teacher-detached: the KL gradient is added only to the
student logits slice.

### `SeparateTeacher(hidden, weight)`

`SeparateTeacher` is the separate-head mode. Student and teacher logits are
created with different projection weights:

```text
student logits = student_hidden @ weight.T
teacher logits = teacher_hidden @ teacher_weight.T
```

This mode maps directly to setups where the teacher has its own output head,
such as an external pretrained teacher, an EMA teacher, a teacher with a
different hidden dimension, or a runtime teacher branch.

By default, `teacher_ce_weight=None` resolves to `0.0`, so the mode runs as
student CE + forward KL unless teacher CE is explicitly enabled. The student
weight can receive gradients from student CE and KL. The teacher hidden state
and teacher weight can receive gradients only from the teacher CE branch when
`teacher_ce_weight > 0`; they do not receive KL gradients because the teacher
distribution is detached.

### `PrecomputedTeacher(logits)`

`PrecomputedTeacher` is the cached-logits mode. Teacher logits are supplied
directly as a `[BT, V]` tensor:

```text
teacher logits = supplied_teacher_logits
```

The loss path projects the student hidden states, then concatenates the student
logits with the supplied teacher logits for CE/KL processing. The supplied
teacher logits must not require gradients. This mode is compatible with offline
KD pipelines where teacher logits are precomputed or produced outside the
current loss call.

By default, `teacher_ce_weight=None` resolves to `0.0`. If
`teacher_ce_weight > 0` is used, teacher CE can be reported or included from the
supplied logits, but there is still no teacher gradient path.

## Forward-KL Setups Expressible by These Modes

The following entries should be read as response-based forward-KL usage patterns
that the current teacher modes can express, not as separate KD families
implemented by the library.

| Forward-KL setup | Compatible modes | Notes |
| :--- | :--- | :--- |
| Response-based KD | `TiedTeacher`, `SeparateTeacher`, `PrecomputedTeacher` | All modes operate on output logits and compute forward KL from teacher distribution to student distribution. |
| Offline KD with a frozen pretrained teacher | `SeparateTeacher`, `PrecomputedTeacher`; `TiedTeacher` for shared-head designs | Use `SeparateTeacher` when the frozen teacher is evaluated during training, or `PrecomputedTeacher` when logits are cached. |
| Online KD / self-distillation | `TiedTeacher`; `SeparateTeacher` when the online teacher has a separate head | `TiedTeacher` is compatible with shared-head teacher signals. `SeparateTeacher` can represent an online teacher branch with its own head. |
| Co-distillation / mutual-learning-style training | `SeparateTeacher`; `TiedTeacher` for shared-head variants | `teacher_ce_weight > 0` enables teacher CE gradients. The KL term remains teacher-detached, so the kernel does not implement symmetric mutual KL. |
| EMA teacher / frozen separate head | `SeparateTeacher`; `PrecomputedTeacher` for cached EMA logits | `SeparateTeacher` can use an EMA/frozen teacher head evaluated at runtime. `PrecomputedTeacher` can use logits generated outside the loss call. |

## References and Related Work

This project builds on a small set of established ideas in knowledge
distillation and efficient loss kernels:

- Logit-level knowledge distillation follows the response-based distillation
  setup introduced by Hinton et al. in
  [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
- The optional online softmax path follows the online normalizer idea from
  Milakov and Gimelshein,
  [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) is related in the
  context of memory-efficient Triton loss kernels, especially for Cross Entropy
  kernels. The overlap is in general technical directions such as chunking and
  in-place gradient storage. ORDA applies these directions within a separate
  teacher-student forward-KL distillation lifecycle.
- The demo notebook uses
  [`unsloth/Llama-3.2-1B`](https://huggingface.co/unsloth/Llama-3.2-1B) and
  [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext)
  from Hugging Face.

## Expert Tuning

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

Passing `config=` overrides `profile=`.

`profile="fast"` enables fast math only. INT8 gradient compression and
stochastic rounding are opt-in. Set `stochastic_seed` for reproducible
stochastic quantization.

`max_chunks`: This parameter limits the maximum number of chunks. When set to
`None`, the system automatically computes the limit and only allows the number
of chunks to double, as a power of 2, at most once to reduce chunk size after an
out-of-memory error. The current kernel heuristic splits chunks around 512-1024
rows per chunk based on the fastest results in Tesla T4 tests. That heuristic
may not be right for every GPU, especially with very large BT sizes where the
workload is split into many chunks and the efficiency benefit may taper off.
Users are encouraged to configure `max_chunks` explicitly based on available
VRAM.

`student_ce`, `teacher_ce`, and `kl` are detached reported components. Only
`loss` carries the backward path. `reduction="mean"` divides CE and KL by the
valid-token count; `reduction="sum"` returns summed CE and KL components.

CUDA execution casts the kernel compute path to fp16. The HIP/ROCm path is
intended to cast kernel compute buffers to bf16, but current validation and
performance documentation is scoped to T4/fp16 unless a specific GPU benchmark
states otherwise. Input dtype support does not imply a full fp32 kernel compute
path.

For `PrecomputedTeacher`, teacher logits are treated as supplied constants.
With `teacher_ce_weight > 0`, `teacher_ce` is reported but has no gradient path.

`profile="debug"` selects a numerical-reference/debug configuration.
