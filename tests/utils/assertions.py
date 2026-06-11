from __future__ import annotations


def assert_scalar_close(torch, actual, expected, *, atol: float, rtol: float, name: str) -> None:
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    if not torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol):
        diff = (actual_f - expected_f).abs().item()
        raise AssertionError(
            f"{name} mismatch: actual={actual_f.item():.8g} "
            f"expected={expected_f.item():.8g} abs_diff={diff:.3e} "
            f"atol={atol:.1e} rtol={rtol:.1e}"
        )


def assert_loss_components_match(torch, actual, expected, *, atol: float, rtol: float) -> None:
    _, loss_s, loss_t, kl_loss = actual
    _, ref_s, ref_t, ref_kl = expected[:4]
    assert_scalar_close(torch, loss_s, ref_s, atol=atol, rtol=rtol, name="loss_s")
    assert_scalar_close(torch, loss_t, ref_t, atol=atol, rtol=rtol, name="loss_t")
    assert_scalar_close(torch, kl_loss, ref_kl, atol=atol, rtol=rtol, name="kl_loss")


def assert_grad_cosine(torch, actual, expected, *, min_cos: float, name: str) -> None:
    actual_f = actual.detach().float().flatten().cpu()
    expected_f = expected.detach().float().flatten().cpu()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.unsqueeze(0),
        expected_f.unsqueeze(0),
    ).item()
    if cosine <= min_cos:
        raise AssertionError(f"{name} grad cosine {cosine:.6f} <= {min_cos:.6f}")


def assert_zero_scalar(torch, tensor, *, name: str) -> None:
    if not torch.equal(tensor, torch.zeros_like(tensor)):
        raise AssertionError(f"{name} expected exact zero, got {tensor.detach().float().item():.8g}")


def assert_no_grad(param, *, name: str) -> None:
    if getattr(param, "grad", None) is not None:
        raise AssertionError(f"{name}.grad expected None")


