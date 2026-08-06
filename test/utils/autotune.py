from __future__ import annotations

from orda_kernel._autotune import keys
from orda_kernel._autotune.models import TEACHER_GRAD_CE, TIED_TRAINED


def runtime_values(torch, **overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "mode": TIED_TRAINED,
        "device": torch.device("cpu"),
        "dtype": torch.float16,
        "vocab_size": 8192,
        "bt": 5,
        "max_fused_size": 8192,
        "shape_key": (8,),
        "compute_kl": True,
        "t_is_one": False,
        "compute_teacher_ce": True,
        "compute_student_ce": True,
        "kl_variant": "forward",
        "teacher_grad_policy": TEACHER_GRAD_CE,
        "grad_student_hidden": True,
        "grad_teacher_hidden": True,
        "grad_weight_accum_dtype": "torch.float16",
        "grad_teacher_weight_accum_dtype": None,
    }
    values.update(overrides)
    return values


def identity(torch, **overrides):
    values = runtime_values(torch, **overrides)
    return keys._runtime_key(**values), keys._workload(**values)


def indexed_identity(torch, index: int):
    return identity(torch, shape_key=(8, index + 1))


def runtime_key_from_selection(values):
    excluded = {"autotune", "benchmark_factory", "refresh_autotune"}
    return keys._runtime_key(
        **{name: value for name, value in values.items() if name not in excluded}
    )


def selection_kwargs(torch, **overrides) -> dict[str, object]:
    values = runtime_values(torch)
    values.update(
        autotune=True,
        benchmark_factory=lambda: object(),
    )
    values.update(overrides)
    return values
