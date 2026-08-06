from __future__ import annotations


def validate_positive_int(name: str, value: int, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        operator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be an integer {operator}, got {value!r}")
    return value


def validate_benchmark_args(
    *,
    vocab: int | None = None,
    seq: int | None = None,
    batch: int | None = None,
    warmup: int | None = None,
    update_steps: int | None = None,
    measure_iters: int | None = None,
) -> None:
    for name, value in (("vocab", vocab), ("seq", seq), ("batch", batch)):
        if value is not None:
            validate_positive_int(name, value)
    if warmup is not None:
        validate_positive_int("warmup", warmup, allow_zero=True)
    if update_steps is not None:
        validate_positive_int("update_steps", update_steps)
    if measure_iters is not None:
        validate_positive_int("measure_iters", measure_iters)
