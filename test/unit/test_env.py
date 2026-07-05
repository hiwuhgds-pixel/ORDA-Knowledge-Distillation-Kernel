from __future__ import annotations

import argparse

import pytest
import torch

from utils.env import parse_configs, torch_dtype, validate_cuda_dtype


def test_parse_configs_accepts_comma_separated_batch_sequence_pairs():
    assert parse_configs("2x4, 1x8") == [(2, 4), (1, 8)]


@pytest.mark.parametrize("raw", ["", "2", "2xa", "0x4", "2x0"])
def test_parse_configs_rejects_invalid_values(raw: str):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_configs(raw)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fp16", torch.float16),
        ("half", torch.float16),
        ("bf16", torch.bfloat16),
        ("fp32", torch.float32),
    ],
)
def test_torch_dtype_aliases(name: str, expected):
    assert torch_dtype(torch, name) is expected


def test_torch_dtype_rejects_unknown_alias():
    with pytest.raises(argparse.ArgumentTypeError, match="Unsupported dtype"):
        torch_dtype(torch, "int8")


def test_validate_cuda_dtype_accepts_fp32_on_any_machine():
    validate_cuda_dtype(torch, torch.float32)
