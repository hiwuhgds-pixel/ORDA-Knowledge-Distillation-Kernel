from __future__ import annotations

import torch

from ..config.availability import (
    train_modes,
)
from ..config.profiles import TrainProfile
from ..engine.selection import BenchmarkCase
from ..engine.train import TrainContext, TrainMeasurement, run_isolated
from ..engine.train_runner import (
    TRAIN_SCHEMA,
    print_orda_delta_table,
    print_train_header,
    print_train_summary,
    sweep_shapes,
)
from ..losses.spec import LossSpec
from .specs import (
    ALIGNED_CASE_SPECS,
    CASE_ID_ORDER,
    PRECOMPUTED_CACHE_CASE_SPECS,
    SPECIAL_CASES,
    SpecialCase,
    stable_seed,
    teacher_case_label,
    teacher_case_specs,
    teacher_policy,
    teacher_state_label,
)
from .train import ModelTrainWorkload
from .train_aligned import AlignedTrainWorkload
from .train_precomputed import precomputed_train_case
from .train_tied import TiedTrainWorkload, tied_banner_configs


# ── Case context ─────────────────────────────────────────────────────────────
def _context(
    *,
    profile: TrainProfile,
    vocab: int,
    seq: int,
    mode: str,
    device,
    dtype,
    use_amp: bool,
    use_grad_scaler: bool,
) -> TrainContext:
    return TrainContext(
        vocab=vocab,
        seq=seq,
        batch=profile.batch,
        warmup=profile.warmup,
        update_steps=profile.update_steps,
        dtype=dtype,
        device=device,
        use_amp=use_amp,
        use_grad_scaler=use_grad_scaler,
        mode=mode,
    )


# ── Case builders ────────────────────────────────────────────────────────────
def _separate_case(
    *,
    case_id: str,
    state: str,
    divergence: str,
    profile: TrainProfile,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    use_grad_scaler: bool,
) -> BenchmarkCase:
    policy = teacher_policy(state, divergence)

    def run():
        modes, orda_warn = train_modes()
        spec = LossSpec(
            temperature=profile.temperature,
            kl_divergence=divergence,
            student_ce_weight=1.0,
            teacher_ce_weight=policy.teacher_ce_weight,
            kl_weight=1.0,
        )
        print_train_header(
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            use_grad_scaler=use_grad_scaler,
            case_label=teacher_case_label("Train", "separate", state, divergence),
            batch=profile.batch,
            warmup=profile.warmup,
            update_steps=profile.update_steps,
            passes=profile.passes,
            temperature=profile.temperature,
            teacher=str(profile.teacher_config),
            student=str(profile.student_config),
            kl_divergence=divergence,
            orda_warn=orda_warn,
        )

        def measure(vocab, seq, mode, pass_index) -> TrainMeasurement:
            workload = ModelTrainWorkload(
                teacher_config=profile.teacher_config,
                student_config=profile.student_config,
                spec=spec,
                teacher_requires_grad=policy.teacher_requires_grad,
            )
            return run_isolated(
                workload,
                _context(
                    profile=profile,
                    vocab=vocab,
                    seq=seq,
                    mode=mode,
                    device=device,
                    dtype=dtype,
                    use_amp=use_amp,
                    use_grad_scaler=use_grad_scaler,
                ),
                seed=stable_seed(
                    "train",
                    _teacher_mode(state),
                    divergence,
                    profile.teacher_config.dim,
                    profile.student_config.dim,
                    vocab,
                    seq,
                    profile.batch,
                    pass_index,
                ),
            )

        table = sweep_shapes(
            vocabs=profile.vocabs,
            seqs=profile.seqs,
            modes=modes,
            passes=profile.passes,
            measure=measure,
        )
        print_train_summary(table)
        print_orda_delta_table(table, vocabs=profile.vocabs, seqs=profile.seqs)
        return TRAIN_SCHEMA, table.records

    return BenchmarkCase(
        case_id,
        f"{teacher_state_label(state, 'train')} · {divergence}",
        run,
    )


def _teacher_mode(state: str) -> str:
    if state == "cotrained_teacher_ce":
        return "separate_cotrained_ce"
    return f"separate_{state}"


def _aligned_case(
    *,
    case_id: str,
    objective: str,
    label: str,
    profile: TrainProfile,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    use_grad_scaler: bool,
) -> BenchmarkCase:
    def run():
        modes, orda_warn = train_modes()
        print_train_header(
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            use_grad_scaler=use_grad_scaler,
            case_label=f"aligned span · {label}",
            batch=profile.batch,
            warmup=profile.warmup,
            update_steps=profile.update_steps,
            passes=profile.passes,
            teacher=str(profile.teacher_config),
            student=str(profile.student_config),
            objective=objective,
            orda_warn=orda_warn,
        )

        def measure(vocab, seq, mode, pass_index) -> TrainMeasurement:
            workload = AlignedTrainWorkload(
                student_config=profile.student_config,
                teacher_dim=profile.teacher_config.dim,
                objective=objective,
            )
            return run_isolated(
                workload,
                _context(
                    profile=profile,
                    vocab=vocab,
                    seq=seq,
                    mode=mode,
                    device=device,
                    dtype=dtype,
                    use_amp=use_amp,
                    use_grad_scaler=use_grad_scaler,
                ),
                seed=stable_seed(
                    "train-aligned",
                    objective,
                    profile.student_config.dim,
                    profile.teacher_config.dim,
                    vocab,
                    seq,
                    profile.batch,
                    pass_index,
                ),
            )

        table = sweep_shapes(
            vocabs=profile.vocabs,
            seqs=profile.seqs,
            modes=modes,
            passes=profile.passes,
            measure=measure,
        )
        print_train_summary(table)
        return TRAIN_SCHEMA, table.records

    return BenchmarkCase(case_id, f"aligned span · {label}", run)


def _tied_case(
    *,
    case: SpecialCase,
    profile: TrainProfile,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    use_grad_scaler: bool,
) -> BenchmarkCase:
    def run():
        modes, orda_warn = train_modes()
        teacher_text, student_text = tied_banner_configs(case, profile.teacher_config)
        print_train_header(
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            use_grad_scaler=use_grad_scaler,
            case_label=case.label,
            batch=profile.batch,
            warmup=profile.warmup,
            update_steps=profile.update_steps,
            passes=profile.passes,
            temperature=case.temperature,
            teacher=teacher_text,
            student=student_text,
            kl_divergence=case.kl_divergence,
            orda_warn=orda_warn,
        )

        def measure(vocab, seq, mode, pass_index) -> TrainMeasurement:
            workload = TiedTrainWorkload(
                case=case, teacher_config=profile.teacher_config
            )
            return run_isolated(
                workload,
                _context(
                    profile=profile,
                    vocab=vocab,
                    seq=seq,
                    mode=mode,
                    device=device,
                    dtype=dtype,
                    use_amp=use_amp,
                    use_grad_scaler=use_grad_scaler,
                ),
                seed=stable_seed(
                    "train_tied", case.id, vocab, seq, profile.batch, pass_index
                ),
            )

        table = sweep_shapes(
            vocabs=profile.vocabs,
            seqs=profile.seqs,
            modes=modes,
            passes=profile.passes,
            measure=measure,
        )
        print_train_summary(table)
        print_orda_delta_table(table, vocabs=profile.vocabs, seqs=profile.seqs)
        return TRAIN_SCHEMA, table.records

    return BenchmarkCase(case.id, case.label, run)


# ── Case registry ────────────────────────────────────────────────────────────
def separate_train_cases(
    *, profile, device, dtype, use_amp, use_grad_scaler
) -> list[BenchmarkCase]:
    return [
        _separate_case(
            case_id=spec.case_id,
            state=spec.state,
            divergence=spec.divergence,
            profile=profile,
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            use_grad_scaler=use_grad_scaler,
        )
        for spec in teacher_case_specs()
    ]


def precomputed_train_cases(
    *, profile, device, dtype, use_amp, use_grad_scaler
) -> list[BenchmarkCase]:
    common = dict(
        profile=profile,
        device=device,
        dtype=dtype,
        use_amp=use_amp,
        use_grad_scaler=use_grad_scaler,
    )
    cases = [
        precomputed_train_case(case_id=divergence, divergence=divergence, **common)
        for divergence in CASE_ID_ORDER
    ]
    cases.extend(
        _aligned_case(
            case_id=spec.case_id,
            objective=spec.objective,
            label=spec.label,
            **common,
        )
        for spec in ALIGNED_CASE_SPECS
    )
    cases.extend(
        precomputed_train_case(
            case_id=spec.case_id,
            divergence="forward",
            label=spec.label,
            cache_kind=spec.cache_kind,
            source_labels=("logits",),
            **common,
        )
        for spec in PRECOMPUTED_CACHE_CASE_SPECS
    )
    return cases


def tied_train_cases(
    *, profile, device, dtype, use_amp, use_grad_scaler
) -> list[BenchmarkCase]:
    return [
        _tied_case(
            case=case,
            profile=profile,
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            use_grad_scaler=use_grad_scaler,
        )
        for case in SPECIAL_CASES
    ]


TRAIN_CASE_BUILDERS = {
    "separate": separate_train_cases,
    "precomputed": precomputed_train_cases,
    "tied": tied_train_cases,
}
