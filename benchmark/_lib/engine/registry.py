from __future__ import annotations

from collections.abc import Callable, Sequence

from ..config.profiles import accuracy_profile, benchmark_profile, vram_profile
from ..core.device import get_device, precision_config, require_cuda
from .selection import run_router

BENCHMARKS = ("accuracy", "train", "vram")
FAMILIES = {
    "accuracy": ("tied", "separate", "precomputed"),
    "train": ("tied", "separate", "precomputed"),
    "vram": ("separate", "precomputed"),
}

TITLES = {
    "accuracy": "BENCHMARK: accuracy",
    "train": "BENCHMARK: train",
    "vram": "BENCHMARK: VRAM",
}

PREFLIGHT_LABEL = {"train": "Train", "vram": "VRAM"}


# ── Case sources ─────────────────────────────────────────────────────────────
def _accuracy_cases(family: str, _batch: int | None = None) -> tuple:
    from ..suites.accuracy import accuracy_configs, loss_accuracy_case
    from ..suites.accuracy_precomputed import (
        aligned_accuracy_case,
        precomputed_input_case,
    )
    from ..suites.modes import accuracy_family_modes
    from ..suites.specs import (
        ALIGNED_CASE_SPECS,
        PRECOMPUTED_INPUT_ACCURACY_CASES,
    )

    device = get_device()
    dtype, _, _ = precision_config(device)
    profile = benchmark_profile(device)
    accuracy = accuracy_profile(profile)
    configs = accuracy_configs(profile, accuracy, family=family)

    cases = [
        loss_accuracy_case(
            mode=mode,
            profile=profile,
            configs=configs,
            temperature=accuracy.temperature,
            device=device,
            dtype=dtype,
        )
        for mode in accuracy_family_modes(family)
    ]
    if family == "precomputed":
        cases.extend(
            aligned_accuracy_case(
                case_id=spec.case_id,
                objective=spec.objective,
                label=spec.label,
                configs=configs,
                device=device,
                dtype=dtype,
            )
            for spec in ALIGNED_CASE_SPECS
        )
        cases.extend(
            precomputed_input_case(
                case_id=case_id,
                label=label,
                device=device,
                dtype=dtype,
                temperature=accuracy.temperature,
            )
            for case_id, label in PRECOMPUTED_INPUT_ACCURACY_CASES
        )
    return device, cases


def _train_cases(family: str, batch: int | None = None) -> tuple:
    from ..suites.train_cases import TRAIN_CASE_BUILDERS

    device = get_device()
    dtype, use_amp, use_grad_scaler = precision_config(device)
    profile = benchmark_profile(device, batch=batch)
    cases = TRAIN_CASE_BUILDERS[family](
        profile=profile,
        device=device,
        dtype=dtype,
        use_amp=use_amp,
        use_grad_scaler=use_grad_scaler,
    )
    return device, cases


def _vram_cases(family: str, batch: int | None = None) -> tuple:
    from ..suites.vram_cases import VRAM_CASE_BUILDERS

    device = get_device()
    dtype, _, _ = precision_config(device)
    profile = vram_profile(device, batch=batch)
    cases = VRAM_CASE_BUILDERS[family](profile=profile, device=device, dtype=dtype)
    return device, cases


CASE_SOURCES: dict[str, Callable[[str, int | None], tuple]] = {
    "accuracy": _accuracy_cases,
    "train": _train_cases,
    "vram": _vram_cases,
}


# ── Router dispatch ──────────────────────────────────────────────────────────
def run(benchmark: str, family: str, argv: Sequence[str] | None = None) -> int:
    if benchmark not in CASE_SOURCES:
        raise ValueError(
            f"Unknown benchmark: {benchmark!r}; expected one of {list(BENCHMARKS)}"
        )
    allowed_families = FAMILIES[benchmark]
    if family not in allowed_families:
        raise ValueError(
            f"Unknown {benchmark} family: {family!r}; "
            f"expected one of {list(allowed_families)}"
        )
    label = PREFLIGHT_LABEL.get(benchmark)
    return run_router(
        router=f"{benchmark}_{family}",
        title=TITLES[benchmark],
        case_factory=lambda batch: CASE_SOURCES[benchmark](family, batch),
        supports_batch=benchmark in {"train", "vram"},
        argv=argv,
        preflight=(
            None if label is None else lambda device: require_cuda(device, label)
        ),
    )
