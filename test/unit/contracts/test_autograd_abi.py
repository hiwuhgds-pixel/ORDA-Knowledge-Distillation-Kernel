"""Static autograd-ABI contract for the eight distillation Functions.

These run on CPU (no CUDA needed): they assert the positional gradient contract
that ``backward`` must honor — one gradient per ``forward`` input, in order, with
live gradients only at the documented positions. A mismatch here means a kernel
would misroute or drop a gradient on GPU, which the CUDA suite cannot catch as
cheaply.
"""

from __future__ import annotations

import pytest

from orda_kernel.ops._common import forward_input_count, input_gradient_tuple
from orda_kernel.ops._kernels.precomputed.aligned_span import (
    PrecomputedAlignedSpanFunction,
)
from orda_kernel.ops._kernels.precomputed.tokenwise import PrecomputedCEFunction
from orda_kernel.ops._kernels.separate.cotrained import SeparateCotrainedCEFunction
from orda_kernel.ops._kernels.separate.frozen import SeparateFrozenCEFunction
from orda_kernel.ops._kernels.separate.trained import SeparateTrainedCEFunction
from orda_kernel.ops._kernels.tied.cotrained import TiedCotrainedCEFunction
from orda_kernel.ops._kernels.tied.frozen import TiedFrozenCEFunction
from orda_kernel.ops._kernels.tied.trained import TiedTrainedCEFunction

# (Function, expected input count, positions that carry a live gradient)
_CONTRACTS = [
    (TiedFrozenCEFunction, 7, (0, 2)),
    (TiedTrainedCEFunction, 7, (0, 1, 2)),
    (TiedCotrainedCEFunction, 7, (0, 1, 2)),
    (PrecomputedCEFunction, 9, (0, 1)),
    (PrecomputedAlignedSpanFunction, 8, (0, 1)),
    (SeparateFrozenCEFunction, 8, (0, 2)),
    (SeparateTrainedCEFunction, 8, (0, 1, 2, 3)),
    (SeparateCotrainedCEFunction, 8, (0, 1, 2, 3)),
]


@pytest.mark.parametrize(
    ("function", "expected_count", "_live"),
    _CONTRACTS,
    ids=lambda value: getattr(value, "__name__", value),
)
def test_declared_input_count_matches_forward_signature(
    function, expected_count, _live
):
    assert function.INPUT_COUNT == expected_count
    assert forward_input_count(function) == expected_count


@pytest.mark.parametrize(
    ("function", "expected_count", "live"),
    _CONTRACTS,
    ids=lambda value: getattr(value, "__name__", value),
)
def test_gradient_tuple_places_live_positions_only(function, expected_count, live):
    sentinels = {index: object() for index in live}
    grads = input_gradient_tuple(
        function.INPUT_COUNT, *((index, sentinels[index]) for index in live)
    )
    assert len(grads) == expected_count
    for index in range(expected_count):
        if index in live:
            assert grads[index] is sentinels[index]
        else:
            assert grads[index] is None
