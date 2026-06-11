from __future__ import annotations

from tests.utils.env import set_seed


def _target(torch, *, BT: int, V: int, device: str, ignore_index: int | None):
    target = torch.randint(0, V, (BT,), device=device)
    if ignore_index is not None and BT > 0:
        target[:: max(1, BT // 4)] = ignore_index
    return target


def make_tied_inputs(
    torch,
    *,
    BT: int,
    H: int,
    V: int,
    dtype,
    seed: int,
    device: str = "cuda",
    ignore_index: int | None = -100,
):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    ht = (torch.randn(BT, H, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    weight = (torch.randn(V, H, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    return hs, ht, weight, _target(torch, BT=BT, V=V, device=device, ignore_index=ignore_index)


def make_separate_inputs(
    torch,
    *,
    BT: int,
    Hs: int,
    Ht: int,
    V: int,
    dtype,
    seed: int,
    device: str = "cuda",
    ignore_index: int | None = -100,
    teacher_requires_grad: bool = False,
    weight_teacher_requires_grad: bool = False,
):
    set_seed(torch, seed)
    hs = (torch.randn(BT, Hs, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    ht = torch.randn(BT, Ht, device=device, dtype=dtype) * 0.1
    if teacher_requires_grad:
        ht = ht.requires_grad_(True)
    weight = (torch.randn(V, Hs, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    weight_teacher = torch.randn(V, Ht, device=device, dtype=dtype) * 0.1
    if weight_teacher_requires_grad:
        weight_teacher = weight_teacher.requires_grad_(True)
    target = _target(torch, BT=BT, V=V, device=device, ignore_index=ignore_index)
    return hs, ht, weight, weight_teacher, target


def make_precomputed_inputs(
    torch,
    *,
    BT: int,
    H: int,
    V: int,
    dtype,
    seed: int,
    device: str = "cuda",
    ignore_index: int | None = -100,
):
    set_seed(torch, seed)
    hs = (torch.randn(BT, H, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    weight = (torch.randn(V, H, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    logits_teacher = torch.randn(BT, V, device=device, dtype=dtype) * 0.1
    target = _target(torch, BT=BT, V=V, device=device, ignore_index=ignore_index)
    return hs, weight, logits_teacher, target


