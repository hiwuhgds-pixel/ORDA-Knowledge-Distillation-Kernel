"""Public namespace contracts for precomputed-input utilities."""

from __future__ import annotations

import importlib
import inspect

import pytest

import orda_kernel
from orda_kernel.ops import precomputed_inputs

_BUILDERS = {
    "build_exact_token_map": ("student_vocab", "teacher_vocab"),
    "build_precomputed_span_teacher": (
        "teacher_hidden",
        "teacher_weight",
        "teacher_labels",
        "teacher_ranges",
        "ignore_index",
    ),
    "project_teacher_logits": (
        "teacher_logits",
        "token_map",
        "student_vocab_size",
        "unmatched",
        "temperature",
    ),
    "mix_teacher_logits": ("logits_list", "weights", "mode"),
    "generalized_jensen_shannon": ("logits_list", "weights"),
}


def test_precomputed_input_builders_have_one_public_namespace_and_root_reexports():
    assert set(precomputed_inputs.__all__) == {
        "aligned_span_distillation_loss",
        *_BUILDERS,
    }
    assert (
        precomputed_inputs.aligned_span_distillation_loss
        is orda_kernel.aligned_span_distillation_loss
    )
    for name, parameter_names in _BUILDERS.items():
        namespace_builder = getattr(precomputed_inputs, name)
        root_builder = getattr(orda_kernel, name)

        assert root_builder is namespace_builder
        signature = inspect.signature(namespace_builder)
        assert tuple(signature.parameters) == parameter_names
        assert inspect.signature(root_builder) == signature


@pytest.mark.parametrize(
    "legacy_module",
    [
        "orda_kernel.ops.cross_vocab",
        "orda_kernel.ops.multi_teacher",
        "orda_kernel.ops.aligned_span",
        "orda_kernel.ops._aligned_span_math",
    ],
)
def test_legacy_precomputed_builder_modules_are_removed_without_shims(
    legacy_module: str,
):
    with pytest.raises(
        ModuleNotFoundError, match=legacy_module.rsplit(".", maxsplit=1)[-1]
    ):
        importlib.import_module(legacy_module)
