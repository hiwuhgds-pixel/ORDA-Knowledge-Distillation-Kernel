"""Use an fp64 oracle so precomputed-input checks validate values, not backends."""

from __future__ import annotations

import torch

from ..config.availability import ORDA_IMPORT_ERROR, runtime_available
from ..core.console import print_banner, print_config
from ..core.device import gpu_name, seed_all, sync
from ..core.oom import is_oom_error
from ..engine.accuracy import (
    Gradients,
    backward_with_grad_scaler,
    dtype_text,
    grad_scaler_for,
    gradients_to_cpu,
    normalize_dtype,
    release_case_memory,
)
from ..engine.selection import BenchmarkCase
from ..losses.aligned import aligned_inputs, make_aligned_loss_fn
from ..losses.divergence import kl_term
from ..losses.precomputed import precomputed_cross_vocab_logits
from ..report.render import print_record_table
from .accuracy import (
    ACCURACY_BACKENDS,
    ACCURACY_SCHEMA,
    ORDA_BACKEND_NUM_CHUNKS,
    PRECOMPUTED_INPUT_SCHEMA,
    REFERENCE_BACKEND,
    _metrics,
    _oom_metrics,
    _record,
    _reference_metrics,
    print_config_group,
)


# ── Aligned span ─────────────────────────────────────────────────────────────
def _aligned_case_inputs(config: dict, *, device: torch.device, seed: int):
    seed_all(device, seed)
    return aligned_inputs(
        config["BT"],
        config["V"],
        config["Hs"],
        teacher_dim=config["Ht"],
        dtype=torch.float32,
        device=device,
        hidden_requires_grad=True,
        hidden_scale=0.1,
        weight_scale=0.1,
        student_span_length=2,
        teacher_span_lengths=(1,),
    )


def _aligned_value_grad(
    inputs,
    *,
    objective: str,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    grad_scaler,
):
    hidden, weight, labels, student_ranges, teacher_ranges, teacher = inputs
    num_chunks = ORDA_BACKEND_NUM_CHUNKS.get(mode)
    loss_fn = make_aligned_loss_fn(
        objective,
        "orda" if num_chunks is not None else mode,
        include_ce=False,
        num_chunks=num_chunks,
        autotune=False if num_chunks is not None else None,
        grad_scaler=grad_scaler if num_chunks is not None else None,
    )
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        loss, _, span = loss_fn(
            hidden, weight, labels, student_ranges, teacher_ranges, teacher
        )
    grad_scale = backward_with_grad_scaler(loss, grad_scaler)
    gradients = Gradients(
        ("student_hidden", "student_weight"),
        (hidden.grad.detach(), weight.grad.detach()),
        grad_scale,
    )
    return span.detach(), gradients


def measure_aligned_records(
    *,
    objective: str,
    configs: list[dict],
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict]:
    records: list[dict] = []
    dtype_label = normalize_dtype(dtype)
    case_id = f"aligned-span-{objective}"
    all_backends = (REFERENCE_BACKEND, *ACCURACY_BACKENDS)

    for index, config in enumerate(configs):
        seed = 30_000 + index
        try:
            expected, expected_grad = _aligned_value_grad(
                _aligned_case_inputs(config, device=device, seed=seed),
                objective=objective,
                mode="torch",
                dtype=torch.float32,
                device=device,
                use_amp=False,
                grad_scaler=None,
            )
            expected = expected.double().cpu()
            expected_grad = gradients_to_cpu(expected_grad)
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            for backend in all_backends:
                records.append(
                    _record(
                        config, case_id, "span", dtype_label, backend, _oom_metrics()
                    )
                )
            continue
        finally:
            release_case_memory(device)

        records.append(
            _record(
                config,
                case_id,
                "span",
                dtype_label,
                REFERENCE_BACKEND,
                _reference_metrics(expected, expected_grad),
            )
        )
        for backend in ACCURACY_BACKENDS:
            actual = gradients = None
            try:
                actual, gradients = _aligned_value_grad(
                    _aligned_case_inputs(config, device=device, seed=seed),
                    objective=objective,
                    mode=backend,
                    dtype=dtype,
                    device=device,
                    use_amp=True,
                    grad_scaler=grad_scaler_for(device, dtype),
                )
                metrics = _metrics(actual, gradients, expected, expected_grad)
            except Exception as exc:
                if not is_oom_error(exc):
                    raise
                metrics = _oom_metrics()
            finally:
                actual = gradients = None
                release_case_memory(device)
            records.append(
                _record(config, case_id, "span", dtype_label, backend, metrics)
            )
        del expected, expected_grad
    return records


def aligned_accuracy_case(
    *,
    case_id: str,
    objective: str,
    label: str,
    configs: list[dict],
    device: torch.device,
    dtype: torch.dtype,
) -> BenchmarkCase:
    def run():
        print_banner("BENCHMARK: accuracy")
        print_config(
            {
                "device": str(device),
                "gpu": gpu_name(device),
                "dtype": dtype_text(dtype),
                "baseline": "fp32",
                "component": "span",
                "objective": objective,
                "case": f"aligned span · {label}",
                "vocab": str(configs[0]["V"]),
                "dim": str(configs[0]["Hs"]),
            }
        )
        if not runtime_available(device):
            return None, None
        records = measure_aligned_records(
            objective=objective, configs=configs, dtype=dtype, device=device
        )
        for config in configs:
            print_config_group(config, {"span": records})
        return ACCURACY_SCHEMA, records

    return BenchmarkCase(case_id, f"aligned span · {label}", run)


# ── Precomputed-input correctness ────────────────────────────────────────────
def _error_record(check: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(
            f"{check} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    actual64 = actual.detach().double()
    expected64 = expected.detach().double()
    finite = torch.isfinite(actual64) & torch.isfinite(expected64)
    mismatch = (
        (torch.isfinite(actual64) != torch.isfinite(expected64))
        | (torch.isneginf(actual64) != torch.isneginf(expected64))
        | (torch.isposinf(actual64) != torch.isposinf(expected64))
        | torch.isnan(actual64)
        | torch.isnan(expected64)
    ).sum()
    if bool(finite.any()):
        difference = (actual64[finite] - expected64[finite]).abs()
        max_abs = float(difference.max())
        max_rel = float(
            (difference / expected64[finite].abs().clamp_min(1.0e-12)).max()
        )
    else:
        max_abs = max_rel = 0.0
    shape = "scalar" if actual.ndim == 0 else "x".join(map(str, actual.shape))
    return dict(
        check=check,
        shape=shape,
        nonfinite_mismatch=int(mismatch),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
    )


def _projection_records(device: torch.device) -> list[dict]:
    from orda_kernel import build_exact_token_map, project_teacher_logits

    seed_all(device, 42_000)
    student_vocab = [f"token_{index}" for index in range(257)]
    teacher_vocab = ["teacher_only", *student_vocab[:171], student_vocab[0]]
    token_map = build_exact_token_map(student_vocab, teacher_vocab)
    expected_map = torch.tensor([-1, *range(171), 0], dtype=torch.int64)
    logits = torch.randn(11, len(teacher_vocab), dtype=torch.float64, device=device)
    actual = project_teacher_logits(logits, token_map, len(student_vocab))
    mapped = token_map.to(device)
    matched = mapped >= 0
    probabilities = torch.softmax(logits[:, matched], dim=-1)
    projected = torch.zeros(
        logits.shape[0], len(student_vocab), dtype=logits.dtype, device=device
    )
    projected.index_add_(1, mapped[matched], probabilities)
    expected = torch.where(
        projected > 0,
        projected.log(),
        torch.full_like(projected, float("-inf")),
    )
    return [
        _error_record("token_map", token_map, expected_map),
        _error_record("projected_logits", actual, expected),
    ]


def _multi_teacher_records(mode: str, device: torch.device) -> list[dict]:
    from orda_kernel import generalized_jensen_shannon, mix_teacher_logits

    seed_all(device, {"arithmetic": 42_100, "geometric": 42_200, "gjs": 42_300}[mode])
    first = torch.randn(17, 29, dtype=torch.float64, device=device)
    second = torch.randn(17, 29, dtype=torch.float64, device=device)
    weights = torch.tensor([0.4, 0.6], dtype=torch.float64, device=device)
    log_first = torch.log_softmax(first, dim=-1)
    log_second = torch.log_softmax(second, dim=-1)
    if mode == "gjs":
        actual = generalized_jensen_shannon((first, second), weights)
        mixture = torch.logaddexp(
            log_first + weights[0].log(), log_second + weights[1].log()
        )
        expected = weights[0] * (log_first.exp() * (log_first - mixture)).sum(
            dim=-1
        ) + weights[1] * (log_second.exp() * (log_second - mixture)).sum(dim=-1)
    else:
        actual = mix_teacher_logits((first, second), weights, mode=mode)
        if mode == "arithmetic":
            expected = torch.logaddexp(
                log_first + weights[0].log(), log_second + weights[1].log()
            )
        else:
            expected = weights[0] * log_first + weights[1] * log_second
    return [_error_record(mode, actual, expected)]


def _span_cache_records(device: torch.device) -> list[dict]:
    from orda_kernel import build_precomputed_span_teacher

    seed_all(device, 42_400)
    hidden = torch.randn(13, 7, dtype=torch.float64, device=device)
    weight = torch.randn(19, 7, dtype=torch.float64, device=device)
    labels = torch.randint(0, 19, (13,), device=device)
    ranges = torch.tensor([[0, 2], [2, 7], [7, 13]], device=device)
    teacher = build_precomputed_span_teacher(hidden, weight, labels, ranges)
    log_probs = torch.log_softmax(hidden @ weight.t(), dim=-1)
    token_log_probs = log_probs.gather(1, labels[:, None]).squeeze(1)
    expected = torch.stack(
        [token_log_probs[start:end].sum() for start, end in ranges.tolist()]
    )
    return [_error_record("span_log_probs", teacher.span_log_probs, expected)]


def _aligned_live_records(device: torch.device) -> list[dict]:
    from orda_kernel import (
        AlignedLikelihoodConfig,
        AlignedSpanPairs,
        CrossVocabTeacher,
        KernelConfig,
        aligned_span_distillation_loss,
        build_precomputed_span_teacher,
    )

    seed_all(device, 42_500)
    student_hidden = torch.randn(10, 7, device=device) * 0.1
    student_weight = torch.randn(19, 7, device=device) * 0.1
    student_labels = torch.randint(0, 19, (10,), device=device)
    teacher_hidden = torch.randn(12, 9, device=device) * 0.1
    teacher_weight = torch.randn(13, 9, device=device) * 0.1
    teacher_labels = torch.randint(0, 13, (12,), device=device)
    student_ranges = torch.tensor([[0, 3], [3, 7], [7, 10]], device=device)
    teacher_ranges = torch.tensor([[0, 2], [2, 7], [7, 12]], device=device)
    pairs = AlignedSpanPairs(student_ranges, teacher_ranges)
    live = CrossVocabTeacher(teacher_hidden, teacher_weight, teacher_labels)
    cached = build_precomputed_span_teacher(
        teacher_hidden, teacher_weight, teacher_labels, teacher_ranges
    )
    kernel_config = KernelConfig(
        num_chunks=ORDA_BACKEND_NUM_CHUNKS["orda (32 chunk)"], autotune=False
    )

    def evaluate(teacher, objective: str):
        hidden = student_hidden.detach().clone().requires_grad_(True)
        weight = student_weight.detach().clone().requires_grad_(True)
        out = aligned_span_distillation_loss(
            hidden,
            weight,
            student_labels,
            teacher,
            pairs,
            AlignedLikelihoodConfig(student_ce_weight=0.0, objective=objective),
            kernel_config=kernel_config,
        )
        out.loss.backward()
        gradients = torch.cat((hidden.grad.flatten(), weight.grad.flatten()))
        return out.loss, gradients

    records = []
    for objective in ("binary_ce", "binary_kl"):
        live_loss, live_grad = evaluate(live, objective)
        cached_loss, cached_grad = evaluate(cached, objective)
        records.append(_error_record(f"{objective}_loss", live_loss, cached_loss))
        records.append(_error_record(f"{objective}_grad", live_grad, cached_grad))
    return records


def _cross_vocab_loss_records(
    device: torch.device, dtype: torch.dtype, temperature: float
) -> list[dict]:
    from orda_kernel import KernelConfig, PrecomputedTeacher, distillation_loss

    seed_all(device, 41_000)
    n_rows, vocab, hidden_dim = 128, 512, 64
    hidden_base = torch.randn(n_rows, hidden_dim, device=device) * 0.1
    weight_base = torch.randn(vocab, hidden_dim, device=device) * 0.1
    labels = torch.randint(0, vocab, (n_rows,), device=device)
    teacher_logits = precomputed_cross_vocab_logits(
        n_rows, vocab, dtype=dtype, device=device
    )

    expected_hidden = hidden_base.detach().clone().requires_grad_(True)
    expected_weight = weight_base.detach().clone().requires_grad_(True)
    expected_logits = expected_hidden @ expected_weight.t()
    expected_loss = torch.nn.functional.cross_entropy(expected_logits, labels)
    expected_loss = expected_loss + kl_term(
        expected_logits,
        teacher_logits,
        temperature=temperature,
        divergence="forward",
        labels=labels,
        early_row_scaling=True,
    )
    expected_loss.backward()
    expected_grad = torch.cat(
        (expected_hidden.grad.flatten(), expected_weight.grad.flatten())
    )

    actual_hidden = hidden_base.detach().requires_grad_(True)
    actual_weight = weight_base.detach().requires_grad_(True)
    grad_scaler = grad_scaler_for(device, dtype)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
        actual_loss = distillation_loss(
            actual_hidden,
            actual_weight,
            labels,
            PrecomputedTeacher(logits=teacher_logits),
            student_ce_weight=1.0,
            teacher_ce_weight=0.0,
            kl_weight=1.0,
            kl_temperature=temperature,
            kl_divergence="forward",
            backend="triton",
            config=KernelConfig(
                num_chunks=ORDA_BACKEND_NUM_CHUNKS["orda (32 chunk)"], autotune=False
            ),
            grad_scaler=grad_scaler,
        ).loss
    grad_scale = backward_with_grad_scaler(actual_loss, grad_scaler)
    actual_grad = (
        torch.cat(
            (
                actual_hidden.grad.float().flatten(),
                actual_weight.grad.float().flatten(),
            )
        )
        / grad_scale
    )
    return [
        _error_record("loss", actual_loss, expected_loss),
        _error_record("student_grad", actual_grad, expected_grad),
    ]


def precomputed_input_records(
    case_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
) -> list[dict]:
    if case_id == "cross_vocab_projection":
        return _projection_records(device)
    if case_id == "cross_vocab_end_to_end_precomputed_loss":
        return _cross_vocab_loss_records(device, dtype, temperature)
    if case_id == "aligned_span_live_cross_vocab":
        return _aligned_live_records(device)
    if case_id == "precomputed_span_teacher_cache_build":
        return _span_cache_records(device)
    if case_id == "multi_teacher_gjs":
        return _multi_teacher_records("gjs", device)
    if case_id.startswith("multi_teacher_"):
        return _multi_teacher_records(case_id.removeprefix("multi_teacher_"), device)
    raise ValueError(f"Unknown precomputed-input accuracy case: {case_id}")


def _precomputed_input_dtype_text(case_id: str, dtype: torch.dtype) -> str:
    if case_id == "cross_vocab_end_to_end_precomputed_loss":
        return dtype_text(dtype)
    if case_id == "aligned_span_live_cross_vocab":
        return "fp32"
    return "fp64 reference"


def precomputed_input_case(
    *,
    case_id: str,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
) -> BenchmarkCase:
    def run():
        print_banner("BENCHMARK: accuracy")
        print_config(
            {
                "device": str(device),
                "gpu": gpu_name(device),
                "dtype": _precomputed_input_dtype_text(case_id, dtype),
                "case": case_id,
            }
        )
        if ORDA_IMPORT_ERROR is not None:
            print(f"[WARN] orda_kernel import failed ({ORDA_IMPORT_ERROR})")
            return None, None
        if case_id == "cross_vocab_end_to_end_precomputed_loss" and not (
            runtime_available(device)
        ):
            return None, None
        records = precomputed_input_records(
            case_id, device=device, dtype=dtype, temperature=temperature
        )
        sync(device)
        print_record_table(PRECOMPUTED_INPUT_SCHEMA, records)
        return PRECOMPUTED_INPUT_SCHEMA, records

    return BenchmarkCase(case_id, label, run)
