from __future__ import annotations

import torch

from ..config.availability import backend_modes
from ..config.profiles import VramProfile
from ..engine.selection import BenchmarkCase
from ..engine.vram import relaxed_recompile_limit
from ..engine.vram_runner import (
    PRECOMPUTED_VRAM_SCHEMA,
    VRAM_SCHEMA,
    VramSweep,
    print_vram_header,
    run_sweep,
)
from ..losses.spec import LossSpec
from .specs import (
    ALIGNED_CASE_SPECS,
    CASE_ID_ORDER,
    PRECOMPUTED_CACHE_CASE_SPECS,
    teacher_case_label,
    teacher_case_specs,
    teacher_policy,
    teacher_state_label,
)
from .vram import (
    AlignedVramWorkload,
    PrecomputedVramWorkload,
    SeparateVramWorkload,
    shape_for,
)


# ── Case builders ────────────────────────────────────────────────────────────
def _separate_case(
    *,
    case_id: str,
    state: str,
    divergence: str,
    profile: VramProfile,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchmarkCase:
    policy = teacher_policy(state, divergence)

    def run():
        modes, orda_warn = backend_modes()
        spec = LossSpec(
            temperature=profile.temperature,
            kl_divergence=divergence,
            student_ce_weight=1.0,
            teacher_ce_weight=policy.teacher_ce_weight,
            kl_weight=1.0,
        )
        print_vram_header(
            device=device,
            dtype=dtype,
            profile=profile,
            case_label=teacher_case_label(
                "VRAM backend", "separate", state, divergence
            ),
            temperature=profile.temperature,
            orda_warn=orda_warn,
            kl_divergence=divergence if divergence != "forward" else None,
            unequal_dim_pairs=profile.unequal_dim_pairs,
        )

        def make_workload(*, vocab, seq, dim, teacher_dim, mode):
            return SeparateVramWorkload(
                shape=shape_for(
                    profile,
                    vocab=vocab,
                    seq=seq,
                    dim=dim,
                    teacher_dim=teacher_dim,
                    dtype=dtype,
                    device=device,
                ),
                spec=spec,
                mode=mode,
                num_chunks=profile.num_chunks,
                teacher_requires_grad=policy.teacher_requires_grad,
            )

        sweep = VramSweep(
            profile=profile,
            device=device,
            dtype=dtype,
            modes=modes,
            make_workload=make_workload,
            seed_parts=lambda *, dim, teacher_dim, vocab, seq: (
                "vram",
                "separate",
                divergence,
                "standard",
                dim,
                teacher_dim,
                vocab,
                seq,
            ),
            torch_mode="torch",
            compile_mode="torch-compile",
            orda_mode="orda",
            dim_pairs=profile.unequal_dim_pairs,
        )
        with relaxed_recompile_limit():
            table = run_sweep(sweep)
        return VRAM_SCHEMA, table.records

    return BenchmarkCase(
        case_id,
        f"{teacher_state_label(state, 'VRAM')} · {divergence}",
        run,
    )


def _precomputed_case(
    *,
    case_id: str,
    divergence: str,
    label: str = "PrecomputedTeacher hidden",
    cache_kind: str = "standard",
    cache_source: str = "hidden",
    profile: VramProfile,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchmarkCase:
    def run():
        modes, orda_warn = backend_modes(cache_source)
        spec = LossSpec(
            temperature=profile.temperature,
            kl_divergence=divergence,
            student_ce_weight=1.0,
            teacher_ce_weight=0.0,
            kl_weight=1.0,
        )
        print_vram_header(
            device=device,
            dtype=dtype,
            profile=profile,
            case_label=f"{label} · {divergence}",
            temperature=profile.temperature,
            orda_warn=orda_warn,
            kl_divergence=divergence if divergence != "forward" else None,
            cache_kind=cache_kind,
            unequal_dim_pairs=profile.unequal_dim_pairs,
        )

        def make_workload(*, vocab, seq, dim, teacher_dim, mode):
            return PrecomputedVramWorkload(
                shape=shape_for(
                    profile,
                    vocab=vocab,
                    seq=seq,
                    dim=dim,
                    teacher_dim=teacher_dim,
                    dtype=dtype,
                    device=device,
                ),
                spec=spec,
                mode=mode,
                num_chunks=profile.num_chunks,
                cache_kind=cache_kind,
            )

        sweep = VramSweep(
            profile=profile,
            device=device,
            dtype=dtype,
            modes=modes,
            make_workload=make_workload,
            seed_parts=lambda *, dim, teacher_dim, vocab, seq: (
                "vram",
                "precomputed",
                divergence,
                cache_kind,
                dim,
                teacher_dim,
                vocab,
                seq,
            ),
            torch_mode=f"torch-{cache_source}",
            compile_mode=f"torch-compile-{cache_source}",
            orda_mode=f"orda-{cache_source}",
            dim_pairs=profile.unequal_dim_pairs,
            reports_cache=True,
            reports_teacher_dim=cache_source == "hidden",
        )
        with relaxed_recompile_limit():
            table = run_sweep(sweep)
        return PRECOMPUTED_VRAM_SCHEMA, table.records

    return BenchmarkCase(case_id, f"{label} · {divergence}", run)


def _aligned_case(
    *,
    case_id: str,
    objective: str,
    label: str,
    profile: VramProfile,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchmarkCase:
    def run():
        modes, orda_warn = backend_modes()
        print_vram_header(
            device=device,
            dtype=dtype,
            profile=profile,
            case_label=f"aligned span · {label}",
            temperature=profile.temperature,
            orda_warn=orda_warn,
        )

        def make_workload(*, vocab, seq, dim, teacher_dim, mode):
            return AlignedVramWorkload(
                shape=shape_for(
                    profile,
                    vocab=vocab,
                    seq=seq,
                    dim=dim,
                    teacher_dim=dim,
                    dtype=dtype,
                    device=device,
                ),
                objective=objective,
                mode=mode,
                num_chunks=profile.num_chunks,
            )

        sweep = VramSweep(
            profile=profile,
            device=device,
            dtype=dtype,
            modes=modes,
            make_workload=make_workload,
            seed_parts=lambda *, dim, teacher_dim, vocab, seq: (
                "vram-aligned",
                objective,
                dim,
                vocab,
                seq,
            ),
            torch_mode="torch",
            compile_mode="torch-compile",
            orda_mode="orda",
            dim_pairs=[(dim, dim) for dim in profile.dims],
        )
        return VRAM_SCHEMA, run_sweep(sweep).records

    return BenchmarkCase(case_id, f"aligned span · {label}", run)


# ── Case registry ────────────────────────────────────────────────────────────
def separate_vram_cases(
    *, profile: VramProfile, device: torch.device, dtype: torch.dtype
) -> list[BenchmarkCase]:
    return [
        _separate_case(
            case_id=spec.case_id,
            state=spec.state,
            divergence=spec.divergence,
            profile=profile,
            device=device,
            dtype=dtype,
        )
        for spec in teacher_case_specs()
    ]


def precomputed_vram_cases(
    *, profile: VramProfile, device: torch.device, dtype: torch.dtype
) -> list[BenchmarkCase]:
    cases = [
        _precomputed_case(
            case_id=divergence,
            divergence=divergence,
            profile=profile,
            device=device,
            dtype=dtype,
        )
        for divergence in CASE_ID_ORDER
    ]
    cases.extend(
        _aligned_case(
            case_id=spec.case_id,
            objective=spec.objective,
            label=spec.label,
            profile=profile,
            device=device,
            dtype=dtype,
        )
        for spec in ALIGNED_CASE_SPECS
    )
    cases.extend(
        _precomputed_case(
            case_id=spec.case_id,
            divergence="forward",
            label=spec.label,
            cache_kind=spec.cache_kind,
            cache_source="logits",
            profile=profile,
            device=device,
            dtype=dtype,
        )
        for spec in PRECOMPUTED_CACHE_CASE_SPECS
    )
    return cases


VRAM_CASE_BUILDERS = {
    "separate": separate_vram_cases,
    "precomputed": precomputed_vram_cases,
}
