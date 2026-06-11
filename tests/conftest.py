from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clear_cuda_cache_after_test():
    yield
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


