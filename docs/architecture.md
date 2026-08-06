# Architecture

`orda_kernel` provides tokenwise and span-aligned distillation APIs with Torch and Triton backends. Policy resolution happens in cold-path Python; vocabulary reduction and logit gradients are handled by one constexpr-specialized Triton kernel. This document describes runtime execution flow. Kernel-level contracts (constexpr axes, pointer layout, autograd ABI, extension rules) are in [designer.md](designer.md); user-facing contract in [reference.md](reference.md) and [variants.md](variants.md).

## Package layout

```text
src/orda_kernel/
├── api.py                         high-level functional API
├── _api_types.py                  public descriptors and configuration
├── _config.py                     profile + KernelConfig merge
├── _validation.py                 shape/dtype/device/label contracts
├── _torch_backend.py              full-logit reference (numerical source of truth)
├── _planning/                     immutable policy and execution plans
│   ├── capabilities.py            product boundary (supported combinations)
│   ├── policy.py                  routing: weights + variant → kernel path
│   └── models.py                  planning vocabulary, KL hyperparameter specs
├── _execution/                    tokenwise/aligned dispatch and OOM recovery
├── _autotune/                     launch selection and persistent cache
└── ops/
    ├── tied_teacher.py             public low-level tied API
    ├── separate_teacher.py         public low-level separate API
    ├── precomputed_teacher.py      public low-level precomputed API
    ├── precomputed_inputs/
    │   ├── aligned_span.py         public cross-vocabulary span API
    │   ├── span_teacher.py         precomputed span-teacher builder
    │   ├── cross_vocab.py          token-map projection builders
    │   ├── multi_teacher.py        multi-teacher mixing builders
    │   └── _aligned_span_math.py   shared FP64 span reductions
    ├── _common.py                  shared tensor/autograd helpers
    └── _kernels/
        ├── ce_kl_fwdbwd.py         canonical Triton CE/KL kernel (only launched kernel)
        ├── _kl_math.py             inline @triton.jit math helpers (never launched)
        ├── tied/{frozen,trained,cotrained}.py
        ├── precomputed/{tokenwise,aligned_span}.py
        └── separate/{frozen,trained,cotrained}.py
```

Each mode file is a vertical slice: benchmark task, chunk loop, GEMMs, autograd Function, and fixed gradient ABI stay together. Only the numerical Triton body is shared.

The three public low-level modules export only their `*_distillation_loss` functions; autograd classes and kernel helpers are private implementation API.

## Public execution flow

1. `distillation_loss` validates tensors, resolves profile/`KernelConfig` merge, resolves KL hyperparameters, and normalizes loss scalars.
2. `build_execution_plan` (or `build_aligned_span_execution_plan`) resolves backend, kernel path, specialization, and chunk policy into an immutable plan. With `backend="auto"`, a policy that no Triton specialization executes re-plans onto the Torch backend and emits the one-time fallback warning.
3. Torch plans call the full-logit reference backend.
4. Tokenwise Triton plans enter `execute_plan`; aligned-span plans enter their dedicated executor. Both pass the plan and a `ChunkAttempt` to the mode slice.
5. Dynamic plans retry with more chunks after recoverable OOM failures and cache the successful partition.

## Teacher modes at runtime

**Tied:** student and teacher share one projection weight. Student and teacher logits are stacked in one temporary buffer.

**Tied cotrained:** mutual/cojeffreys KL under tied layout; teacher hidden receives KL (and optional CE) gradient; shared weight accumulates both stacked halves.

**Separate frozen:** teacher logits materialized chunk-wise, no gradient.

**Separate trained:** teacher CE owns teacher gradients; every teacher-detached KL variant lowers on the Triton path.

**Separate cotrained:** mutual/cojeffreys KL owns teacher gradients, with optional teacher CE contribution.

**Precomputed tokenwise:** teacher logits supplied directly or recomputed from cached hidden+weight; never any teacher gradient.

**Precomputed aligned span:** detached teacher span log-likelihoods drive a two-stage student-only pipeline; a live `CrossVocabTeacher` stays on the Torch reference and can be converted with `build_precomputed_span_teacher`.

`KERNEL_CAPABILITIES` in `_planning/capabilities.py` is the product boundary: the canonical kernel may compile other constexpr combinations, but unsupported mode/variant pairs are rejected before launch. Full routing/eligibility tables are in [variants.md](variants.md).

## Chunking and memory

The Triton backend materializes only chunk-local logits. `ChunkPolicy` controls initial partition and retry limit; a successful dynamic partition is cached by tensor and specialization identity (bounded LRU). Direct low-level APIs use the same execution engine and therefore the same OOM retry behavior.

Public chunk requests use the `ChunkSize` value object (`ChunkSize.nrows(n)` or `ChunkSize.dynamic()` — the complete request type). `num_chunks` and `max_chunks` remain separate controls whose cross-field constraints are validated before planning.

The Torch backend is the numerical reference and materializes full `(BT, vocab)` logits. `backend="auto"` emits one process-wide, thread-safe warning before such a fallback.

## Autotune and cache

Autotune benchmarks full-batch pipelines, not isolated kernel calls; one selection call happens per autograd Function. Aligned-span keeps a distinct pipeline identity; binary KL and binary CE share one winner because their fused stages and logits gradient are identical.

Winners live in an in-memory LRU plus a SQLite WAL database keyed by device fingerprint, workload/pipeline keys, a timed-source AST digest, and generation fencing with expiring cross-process device leases. KL hyperparameters and temperatures are runtime plan scalars — sweeping them never re-benchmarks. Environment variables and cache locations are tabulated in [reference.md §12](reference.md#12-autotune-environment-variables); identity/digest contract is in [designer.md §7](designer.md#7-autotune-identity).

## Multiprocessing

SQLite writes use transactions and WAL. Only one process benchmarks a physical device at a time; lease waiting is bounded, and contention falls back to the latest winner or default launch config instead of blocking training. Schema recovery revalidates the database under a per-store lock before replacing owned files.
