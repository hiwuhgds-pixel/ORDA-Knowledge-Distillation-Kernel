"""Bounded real-GPU coverage for staged full-BT autotune execution paths."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from utils.env import pytest_skip_if_no_cuda_kernel
from utils.matrices import ShardSpec, shard_spec_from_environment

runtime = pytest_skip_if_no_cuda_kernel(pytest)
torch = runtime.torch
F = torch.nn.functional

from orda_kernel import (
    ChunkSize,
    KernelConfig,
    PrecomputedTeacher,
    SeparateTeacher,
    TiedTeacher,
    distillation_loss,
    reset_autotune_cache,
)
from orda_kernel._autotune import tuner as autotune_tuner
from orda_kernel._autotune.benchmark import BenchmarkTask, staged_invocation_bound
from orda_kernel.ops._kernels.tied.trained import _tied_trained_fwdbwd


# ── Bounded supported-path matrix ────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class AutotuneCase:
    ordinal: int
    path: str
    kl_variant: str
    teacher_ce_weight: float | None
    bt: int

    @property
    def case_id(self) -> str:
        tail = "tail" if self.bt == 5 else "full"
        teacher_ce = (
            "default" if self.teacher_ce_weight is None else str(self.teacher_ce_weight)
        )
        return f"{self.ordinal:02d}-{self.path}-{self.kl_variant}-ce{teacher_ce}-{tail}"


_SUPPORTED_PATHS = (
    ("tied_frozen", "forward", 0.0),
    ("tied_trained", "forward", 0.5),
    ("tied_cotrained_kl", "mutual", 0.0),
    ("tied_cotrained_ce_kl", "cojeffreys", 0.5),
    ("separate_trained", "forward", 0.5),
    ("separate_frozen", "forward", 0.0),
    ("separate_cotrained_kl", "mutual", 0.0),
    ("separate_cotrained_ce_kl", "cojeffreys", 0.5),
    ("precomputed", "forward", 0.0),
    ("precomputed", "jeffreys", 0.0),
)
_BT_SHAPES = (5, 6)  # 5 has two full chunks plus a tail under the same config.
_ALL_CASES = tuple(
    AutotuneCase(ordinal, path, variant, teacher_ce_weight, bt)
    for ordinal, ((path, variant, teacher_ce_weight), bt) in enumerate(
        (path_spec, bt) for path_spec in _SUPPORTED_PATHS for bt in _BT_SHAPES
    )
)
_SHARD_SPEC: ShardSpec = shard_spec_from_environment()
_SHARD_CASES = tuple(
    case for case in _ALL_CASES if case.ordinal % _SHARD_SPEC.count == _SHARD_SPEC.index
)
assert {case.ordinal for case in _ALL_CASES} == set(range(len(_ALL_CASES)))


def _make_inputs(case: AutotuneCase, *, vocab_size: int = 8192):
    torch.manual_seed(80_000 + case.ordinal)
    hidden_size = 8
    student_hidden = (
        torch.randn(case.bt, hidden_size, device="cuda", dtype=torch.float16) * 0.1
    ).requires_grad_(True)
    weight = (
        torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float16) * 0.1
    ).requires_grad_(True)
    labels = torch.randint(0, vocab_size, (case.bt,), device="cuda")
    teacher_hidden = (
        torch.randn(case.bt, hidden_size, device="cuda", dtype=torch.float16) * 0.1
    ).requires_grad_(True)
    teacher_weight = (
        torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float16) * 0.1
    ).requires_grad_(True)
    teacher_logits = (
        torch.randn(
            case.bt,
            vocab_size,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.1
    )
    return (
        student_hidden,
        weight,
        labels,
        teacher_hidden,
        teacher_weight,
        teacher_logits,
    )


def _teacher_for_case(
    case: AutotuneCase,
    teacher_hidden,
    teacher_weight,
    teacher_logits,
):
    if case.path == "tied_frozen":
        return TiedTeacher(teacher_hidden, mode="frozen")
    if case.path == "tied_trained":
        return TiedTeacher(teacher_hidden, mode="trained")
    if case.path.startswith("tied_cotrained_"):
        return TiedTeacher(teacher_hidden, mode="cotrained")
    if case.path.startswith("separate_"):
        return SeparateTeacher(teacher_hidden, teacher_weight)
    return PrecomputedTeacher(logits=teacher_logits)


# ── Real staged autotune contract ────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.autotune
@pytest.mark.parametrize("case", _SHARD_CASES, ids=lambda case: case.case_id)
def test_staged_autotune_executes_full_bt_supported_path(case):
    reset_autotune_cache()
    (
        student_hidden,
        weight,
        labels,
        teacher_hidden,
        teacher_weight,
        teacher_logits,
    ) = _make_inputs(case)
    config = KernelConfig(
        chunk_size=ChunkSize.nrows(2),
        max_fused_size=8192,
        autotune=True,
    )
    output = distillation_loss(
        student_hidden,
        weight,
        labels,
        _teacher_for_case(case, teacher_hidden, teacher_weight, teacher_logits),
        teacher_ce_weight=case.teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=1.4,
        kl_divergence=case.kl_variant,
        backend="triton",
        config=config,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert student_hidden.grad is not None
    assert weight.grad is not None


# ── Cold-start full-plan bound ───────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.autotune
def test_staged_autotune_cold_start_stays_within_full_plan_bound(monkeypatch):
    # BT=5 with chunk_size=2 has two full chunks and a one-row tail.
    case = next(
        case for case in _ALL_CASES if case.path == "tied_trained" and case.bt == 5
    )
    reset_autotune_cache()
    (
        student_hidden,
        weight,
        labels,
        teacher_hidden,
        _teacher_weight,
        _teacher_logits,
    ) = _make_inputs(case)
    full_plan_runs = 0
    original_bench_config = autotune_tuner.bench_config

    def count_full_plan_runs(task, config, **kwargs):
        def run(*args):
            nonlocal full_plan_runs
            full_plan_runs += 1
            return task.run(*args)

        # The original CUDA-event benchmark still measures the real task.
        return original_bench_config(
            BenchmarkTask(run=run, prepare_batch=task.prepare_batch),
            config,
            **kwargs,
        )

    monkeypatch.setattr(autotune_tuner, "bench_config", count_full_plan_runs)
    output = distillation_loss(
        student_hidden,
        weight,
        labels,
        TiedTeacher(teacher_hidden),
        teacher_ce_weight=case.teacher_ce_weight,
        kl_weight=0.35,
        kl_temperature=1.4,
        kl_divergence=case.kl_variant,
        backend="triton",
        config=KernelConfig(
            chunk_size=ChunkSize.nrows(2),
            max_fused_size=8192,
            autotune=True,
        ),
    )
    output.loss.backward()

    candidate_count = len(
        autotune_tuner.valid_candidates(8192, 8192, student_hidden.device)
    )
    assert full_plan_runs > 0
    assert full_plan_runs <= staged_invocation_bound(
        candidate_count=candidate_count,
    )


# ── Large-block coverage ─────────────────────────────────────────────────────
@pytest.mark.gpu
@pytest.mark.autotune
@pytest.mark.parametrize("vocab_size,block_size", [(12_000, 16_384), (20_000, 32_768)])
def test_large_block_launch_matches_torch_reference(vocab_size, block_size):
    torch.manual_seed(91_000 + block_size)
    n_rows = 2
    temperature = 1.4
    kl_weight = 0.35
    labels = torch.randint(0, vocab_size, (n_rows,), device="cuda")
    student_logits = torch.randn(n_rows, vocab_size, device="cuda", dtype=torch.float32)
    teacher_logits = torch.randn(n_rows, vocab_size, device="cuda", dtype=torch.float32)
    student_ref = student_logits.detach().clone().requires_grad_(True)
    teacher_ref = teacher_logits.detach().clone().requires_grad_(True)
    student_ce_ref = F.cross_entropy(student_ref, labels, reduction="none")
    teacher_ce_ref = F.cross_entropy(teacher_ref, labels, reduction="none")
    teacher_probs = F.softmax(teacher_ref.detach() / temperature, dim=-1)
    kl_ref = F.kl_div(
        F.log_softmax(student_ref / temperature, dim=-1),
        teacher_probs,
        reduction="none",
    ).sum(dim=-1) * (temperature * temperature)
    (student_ce_ref.sum() + teacher_ce_ref.sum() + kl_weight * kl_ref.sum()).backward()

    logits = torch.cat((student_logits, teacher_logits), dim=0).contiguous()
    student_loss = torch.empty(n_rows, device="cuda", dtype=torch.float32)
    teacher_loss = torch.empty(n_rows, device="cuda", dtype=torch.float32)
    kl_loss = torch.empty(n_rows, device="cuda", dtype=torch.float32)
    _tied_trained_fwdbwd(
        logits,
        labels,
        student_loss,
        kl_loss,
        teacher_loss=teacher_loss,
        kl_weight=kl_weight,
        kl_temperature=temperature,
        n_rows=n_rows,
        student_scale=1.0,
        teacher_scale=1.0,
        max_fused_size=block_size,
        block_size=block_size,
        num_warps=8,
        assume_valid=True,
    )

    assert torch.allclose(student_loss, student_ce_ref, rtol=2e-4, atol=2e-4)
    assert torch.allclose(teacher_loss, teacher_ce_ref, rtol=2e-4, atol=2e-4)
    assert torch.allclose(kl_loss, kl_ref, rtol=5e-4, atol=5e-4)
    assert torch.allclose(logits[:n_rows], student_ref.grad, rtol=5e-4, atol=5e-4)
    assert torch.allclose(logits[n_rows:], teacher_ref.grad, rtol=5e-4, atol=5e-4)


@pytest.mark.gpu
@pytest.mark.autotune
def test_max_autotune_benchmarks_large_block_candidates(monkeypatch):
    case = next(
        case for case in _ALL_CASES if case.path == "tied_trained" and case.bt == 5
    )
    reset_autotune_cache()
    (
        student_hidden,
        weight,
        labels,
        teacher_hidden,
        _teacher_weight,
        _teacher_logits,
    ) = _make_inputs(case, vocab_size=32_768)
    attempted_blocks = set()
    original_bench_config = autotune_tuner.bench_config

    def capture_block(task, config, **kwargs):
        attempted_blocks.add(config.block_size)
        return original_bench_config(task, config, **kwargs)

    monkeypatch.setattr(autotune_tuner, "bench_config", capture_block)
    output = distillation_loss(
        student_hidden,
        weight,
        labels,
        TiedTeacher(teacher_hidden),
        teacher_ce_weight=1.0,
        kl_weight=0.35,
        kl_temperature=1.4,
        backend="triton",
        config=KernelConfig(
            chunk_size=ChunkSize.nrows(2),
            max_fused_size=32_768,
            autotune="Max",
        ),
    )
    output.loss.backward()

    assert {16_384, 32_768} <= attempted_blocks
