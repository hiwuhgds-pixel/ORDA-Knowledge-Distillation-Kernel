from __future__ import annotations

import torch

from ..config.availability import runtime_available
from ..config.profiles import AccuracyProfile, TrainProfile
from ..core.console import print_banner, print_config, print_group_header
from ..core.device import gpu_name, release_memory, sync
from ..core.oom import is_oom_error
from ..engine.accuracy import (
    case_model_inputs,
    dtype_text,
    eager_reference_component,
    grad_metrics,
    grad_scaler_for,
    gradients_to_cpu,
    make_case,
    normalize_dtype,
    orda_component,
    release_case_memory,
    torch_component,
)
from ..engine.selection import BenchmarkCase
from ..report.render import print_record_table
from ..report.schema import Column, TableSchema
from ..suites.modes import MODE_INDEX, accuracy_mode, active_components

# ── Accuracy schemas ─────────────────────────────────────────────────────────
ORDA_BACKEND_NUM_CHUNKS = {
    "orda (1 chunk)": 1,
    "orda (8 chunk)": 8,
    "orda (32 chunk)": 32,
}
TORCH_BACKENDS = ("torch", "torch-compile")
ACCURACY_BACKENDS = (*TORCH_BACKENDS, *ORDA_BACKEND_NUM_CHUNKS)
REFERENCE_BACKEND = "fp32"


def fmt_float(value: object) -> str:
    if isinstance(value, str):
        return value
    numeric = float(value)
    if numeric == 0.0:
        return "0"
    return f"{numeric:.6e}"


def fmt_grad_cos(value: object) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.8f}"


ACCURACY_SCHEMA = TableSchema(
    (
        Column("case", "case"),
        Column("component", "component"),
        Column("config", "config"),
        Column("dtype", "dtype"),
        Column("backend", "backend"),
        Column("value", "value", align_right=True, fmt=fmt_float),
        Column("abs_err", "abs_err", align_right=True, fmt=fmt_float),
        Column("rel_err", "rel_err", align_right=True, fmt=fmt_float),
        Column("grad_cos", "grad_cos", align_right=True, fmt=fmt_grad_cos),
        Column("grad_mean", "grad_mean", align_right=True, fmt=fmt_float),
        Column("grad_max", "grad_max", align_right=True, fmt=fmt_float),
        Column("underflow", "underflow", align_right=True),
    )
)

LOSS_KEYS = ("backend", "value", "abs_err", "rel_err")
GRAD_KEYS = ("backend", "grad_cos", "grad_mean", "grad_max", "underflow")

PRECOMPUTED_INPUT_SCHEMA = TableSchema(
    (
        Column("check", "check"),
        Column("shape", "shape"),
        Column("nonfinite_mismatch", "nonfinite_mismatch", align_right=True),
        Column("max_abs_error", "max_abs_error", align_right=True),
        Column("max_rel_error", "max_rel_error", align_right=True),
    )
)

NA_METRICS = ("NA",) * 6


# ── Case configuration ───────────────────────────────────────────────────────
def accuracy_configs(
    profile: TrainProfile, accuracy: AccuracyProfile, *, family: str
) -> list[dict]:
    from ..core.validation import validate_benchmark_args

    student_config = (
        profile.teacher_config if family == "tied" else profile.student_config
    )
    configs = []
    for seq, batch in accuracy.seq_batches:
        validate_benchmark_args(vocab=accuracy.vocab, seq=seq, batch=batch)
        configs.append(
            dict(
                batch=batch,
                seq=seq,
                BT=batch * seq,
                Hs=student_config.dim,
                Ht=profile.teacher_config.dim,
                V=accuracy.vocab,
            )
        )
    return configs


def fmt_config(config: dict) -> str:
    student_dim, teacher_dim = config["Hs"], config["Ht"]
    dimensions = (
        f"student_dim={student_dim}  teacher_dim={teacher_dim}"
        if student_dim != teacher_dim
        else f"dim={student_dim}"
    )
    return (
        f"vocab={config['V']}  seq={config['seq']}  "
        f"batch={config['batch']}  {dimensions}"
    )


# ── Metric scoring ───────────────────────────────────────────────────────────
def _record(
    config: dict,
    case_id: str,
    component: str,
    dtype_label: str,
    backend: str,
    metrics,
) -> dict:
    value, abs_err, rel_err, grad_cos, grad_mean, grad_max, underflow = metrics
    return dict(
        case=case_id,
        component=component,
        config=fmt_config(config),
        dtype=dtype_label,
        backend=backend,
        value=value,
        abs_err=abs_err,
        rel_err=rel_err,
        grad_cos=grad_cos,
        grad_mean=grad_mean,
        grad_max=grad_max,
        underflow=underflow,
    )


def _oom_metrics() -> tuple:
    return ("OOM", *NA_METRICS)


def _reference_metrics(expected: torch.Tensor, expected_grad) -> tuple:
    nonzero = sum((tensor != 0).sum().item() for tensor in expected_grad.tensors)
    return (
        float(expected.item()),
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        f"0/{nonzero}",
    )


def _metrics(actual: torch.Tensor, actual_grad, expected: torch.Tensor, expected_grad):
    actual = actual.detach().double().cpu()
    actual_grad = gradients_to_cpu(actual_grad)
    abs_error = abs(float(actual.item() - expected.item()))
    denominator = max(abs(float(expected.item())), 1.0e-12)
    grad_cos, grad_mean, grad_max, underflow, underflow_total = grad_metrics(
        expected_grad, actual_grad
    )
    return (
        float(actual.item()),
        abs_error,
        abs_error / denominator,
        float(grad_cos),
        float(grad_mean),
        float(grad_max),
        f"{underflow}/{underflow_total}",
    )


def _component_backend(
    case,
    *,
    backend: str,
    component: str,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    grad_scaler,
):
    if backend in TORCH_BACKENDS:
        return torch_component(
            case,
            component=component,
            dtype=dtype,
            device=device,
            use_amp=use_amp,
            compile_graph=backend == "torch-compile",
            grad_scaler=grad_scaler,
        )
    if backend in ORDA_BACKEND_NUM_CHUNKS:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            return orda_component(
                case,
                component,
                backend="triton",
                num_chunks=ORDA_BACKEND_NUM_CHUNKS[backend],
                grad_scaler=grad_scaler,
            )
    raise ValueError(f"Unknown accuracy backend: {backend}")


# ── Case measurement ─────────────────────────────────────────────────────────
def measure_case_components(
    *,
    mode: str,
    configs: list[dict],
    profile: TrainProfile,
    temperature: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, list[dict]]:
    from orda_kernel._execution.dispatcher import clear_chunk_cache

    spec = accuracy_mode(mode)
    component_records: dict[str, list[dict]] = {
        component: [] for component in active_components(mode)
    }
    dtype_label = normalize_dtype(dtype)
    mode_index = MODE_INDEX[mode]
    all_backends = (REFERENCE_BACKEND, *ACCURACY_BACKENDS)

    for config_index, config in enumerate(configs):
        seed = 10_000 + config_index * 100 + mode_index
        clear_chunk_cache()
        sync(device)
        reference_source = amp_source = None
        try:
            model_inputs = case_model_inputs(
                spec, config, profile=profile, device=device, seed=seed
            )
            reference_source = make_case(
                spec,
                config,
                device=device,
                model_inputs=model_inputs,
                temperature=temperature,
            )
            amp_source = make_case(
                spec,
                config,
                device=device,
                model_inputs=model_inputs,
                temperature=temperature,
                use_amp=True,
            )
            model_inputs = None
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            for component in component_records:
                for backend in all_backends:
                    component_records[component].append(
                        _record(
                            config,
                            mode,
                            component,
                            dtype_label,
                            backend,
                            _oom_metrics(),
                        )
                    )
            reference_source = amp_source = None
            release_case_memory(device)
            continue

        for component in component_records:
            try:
                expected, expected_grad = eager_reference_component(
                    reference_source.reset_gradients(), component=component
                )
                expected = expected.detach().double().cpu()
                expected_grad = gradients_to_cpu(expected_grad)
            except Exception as exc:
                if not is_oom_error(exc):
                    raise
                for backend in all_backends:
                    component_records[component].append(
                        _record(
                            config,
                            mode,
                            component,
                            dtype_label,
                            backend,
                            _oom_metrics(),
                        )
                    )
                release_case_memory(device)
                continue

            component_records[component].append(
                _record(
                    config,
                    mode,
                    component,
                    dtype_label,
                    REFERENCE_BACKEND,
                    _reference_metrics(expected, expected_grad),
                )
            )

            for backend in ACCURACY_BACKENDS:
                actual = actual_grad = None
                try:
                    actual, actual_grad = _component_backend(
                        amp_source.reset_gradients(),
                        backend=backend,
                        component=component,
                        dtype=dtype,
                        device=device,
                        use_amp=True,
                        grad_scaler=grad_scaler_for(device, dtype),
                    )
                    sync(device)
                    metrics = _metrics(actual, actual_grad, expected, expected_grad)
                except Exception as exc:
                    if not is_oom_error(exc):
                        raise
                    metrics = _oom_metrics()
                    release_case_memory(device)
                finally:
                    actual = actual_grad = None
                component_records[component].append(
                    _record(config, mode, component, dtype_label, backend, metrics)
                )
            del expected, expected_grad

        reference_source = amp_source = None
        release_memory(device)
    return component_records


# ── Report rendering ─────────────────────────────────────────────────────────
def print_component_tables(records: list[dict], component: str) -> None:
    print_record_table(
        ACCURACY_SCHEMA.select(*LOSS_KEYS, title=f"{component} · loss"),
        records,
    )
    print_record_table(
        ACCURACY_SCHEMA.select(
            *GRAD_KEYS, title=f"{component} · grad all active tensors"
        ),
        records,
    )


def print_config_group(config: dict, component_records: dict[str, list[dict]]) -> None:
    print_group_header(f"seq={config['seq']}  batch={config['batch']}")
    label = fmt_config(config)
    for component, records in component_records.items():
        print_component_tables(
            [record for record in records if record["config"] == label],
            component.upper(),
        )


def loss_accuracy_case(
    *,
    mode: str,
    profile: TrainProfile,
    configs: list[dict],
    temperature: float,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchmarkCase:
    spec = accuracy_mode(mode)

    def run():
        print_banner("BENCHMARK: accuracy")
        print_config(
            {
                "device": str(device),
                "gpu": gpu_name(device),
                "dtype": dtype_text(dtype),
                "baseline": "fp32",
                "components": "ce, kl",
                "case": mode,
            }
        )
        if not runtime_available(device):
            return None, None
        print(f" vocab       : {configs[0]['V']}")
        if configs[0]["Hs"] != configs[0]["Ht"]:
            print(f" student dim : {configs[0]['Hs']}")
            print(f" teacher dim : {configs[0]['Ht']}")
        else:
            print(f" dim         : {configs[0]['Hs']}")
        component_records = measure_case_components(
            mode=mode,
            configs=configs,
            profile=profile,
            temperature=spec.temperature or temperature,
            dtype=dtype,
            device=device,
        )
        for config in configs:
            print_config_group(config, component_records)
        records = [record for group in component_records.values() for record in group]
        del component_records
        release_case_memory(device)
        return ACCURACY_SCHEMA, records

    return BenchmarkCase(mode, spec.display_label, run)
