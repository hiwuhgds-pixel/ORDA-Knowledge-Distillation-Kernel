from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import orda_ce_kernel
from orda_ce_kernel import api as public_api


def _valid_cpu_inputs():
    student_hidden = torch.randn(4, 8, requires_grad=True)
    teacher_hidden = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(16, 8, requires_grad=True)
    labels = torch.randint(0, 16, (4,))
    return student_hidden, teacher_hidden, weight, labels


def test_public_availability_flag_is_boolean():
    assert isinstance(orda_ce_kernel.is_available(), bool)


def test_public_exports_are_expected_surface():
    expected = {
        "DistillationLoss",
        "DistillationLossOutput",
        "KernelConfig",
        "PrecomputedTeacher",
        "SeparateTeacher",
        "TiedTeacher",
        "distillation_loss",
        "is_available",
    }
    assert set(orda_ce_kernel.__all__) == expected
    for name in expected:
        assert hasattr(orda_ce_kernel, name)
    for internal_name in [
        "enable_fast_math",
        "set_max_chunks",
        "distill_cross_entropy",
        "DistillCrossEntropyLoss",
    ]:
        assert not hasattr(orda_ce_kernel, internal_name)


def test_tied_teacher_torch_backend_returns_named_output_and_backprops():
    hs, ht, weight, labels = _valid_cpu_inputs()
    out = orda_ce_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_ce_kernel.TiedTeacher(ht),
        kd_weight=0.2,
        temperature=1.5,
        backend="torch",
    )

    assert isinstance(out, orda_ce_kernel.DistillationLossOutput)
    loss, student_ce, teacher_ce, kl = out
    assert loss is out.loss
    assert student_ce is out.student_ce
    assert teacher_ce is out.teacher_ce
    assert kl is out.kl
    out.loss.backward()
    assert hs.grad is not None
    assert ht.grad is not None
    assert weight.grad is not None


def test_separate_teacher_default_teacher_ce_weight_is_zero():
    hs = torch.randn(3, 4, requires_grad=True)
    ht = torch.randn(3, 6, requires_grad=True)
    weight = torch.randn(11, 4, requires_grad=True)
    teacher_weight = torch.randn(11, 6, requires_grad=True)
    labels = torch.randint(0, 11, (3,))

    out = orda_ce_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_ce_kernel.SeparateTeacher(ht, teacher_weight),
        kd_weight=0.3,
        backend="torch",
    )
    out.loss.backward()

    assert torch.equal(out.teacher_ce, torch.zeros_like(out.teacher_ce))
    assert teacher_weight.grad is None


def test_precomputed_teacher_rejects_grad_logits():
    hs, _ht, weight, labels = _valid_cpu_inputs()
    logits = torch.randn(labels.shape[0], weight.shape[0], requires_grad=True)
    with pytest.raises(ValueError, match="must not require gradients"):
        orda_ce_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_ce_kernel.PrecomputedTeacher(logits),
            backend="torch",
        )


def test_kernel_config_overrides_profile_and_validates_limits():
    hs, ht, weight, labels = _valid_cpu_inputs()
    cfg = orda_ce_kernel.KernelConfig(max_chunks=0)
    with pytest.raises(ValueError, match="max_chunks"):
        orda_ce_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_ce_kernel.TiedTeacher(ht),
            profile="fast",
            config=cfg,
        )


def test_kernel_config_alias_and_fast_profile_contract():
    fast = public_api._resolve_profile("fast")
    assert fast.fast_math is True
    assert fast.quantize_grad_weight is False
    assert fast.stochastic_rounding is False

    legacy_alias = orda_ce_kernel.KernelConfig(fp32_accumulation=True)
    assert legacy_alias.effective_fp32_grad_weight_accumulation is True
    assert legacy_alias.fp32_accumulation is True
    canonical = orda_ce_kernel.KernelConfig(fp32_grad_weight_accumulation=True)
    assert canonical.fp32_accumulation is True
    legacy_positional = orda_ce_kernel.KernelConfig(True, False, False, False, True)
    assert legacy_positional.effective_fp32_grad_weight_accumulation is True

    with pytest.raises(ValueError, match="cannot disagree"):
        orda_ce_kernel.KernelConfig(
            fp32_grad_weight_accumulation=True,
            fp32_accumulation=False,
        )


def test_kernel_config_validates_stochastic_seed_and_max_fused_size():
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(ValueError, match="stochastic_seed"):
        orda_ce_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_ce_kernel.TiedTeacher(ht),
            config=orda_ce_kernel.KernelConfig(stochastic_seed=-1),
        )
    with pytest.raises(ValueError, match="power of two"):
        orda_ce_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_ce_kernel.TiedTeacher(ht),
            config=orda_ce_kernel.KernelConfig(max_fused_size=100),
        )


def test_bool_labels_are_rejected_before_dispatch():
    hs, ht, weight, _labels = _valid_cpu_inputs()
    labels = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match="integer class-index"):
        orda_ce_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_ce_kernel.TiedTeacher(ht),
            backend="torch",
        )


def _manual_tied_reference(hs, ht, weight, labels, *, kd_weight, temperature, reduction):
    logits_s = hs @ weight.t()
    logits_t = ht @ weight.t()
    mask = labels != -100
    denom = max(int(mask.sum().item()), 1)
    ce_s_all = F.cross_entropy(logits_s, labels, ignore_index=-100, reduction="none")
    ce_t_all = F.cross_entropy(logits_t, labels, ignore_index=-100, reduction="none")
    t = float(temperature)
    kl_all = F.kl_div(
        F.log_softmax(logits_s / t, dim=-1),
        F.softmax(logits_t.detach() / t, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (t * t)
    kl_all = kl_all.masked_fill(~mask, 0.0)
    if reduction == "mean":
        ce_s = ce_s_all.sum() / denom
        ce_t = ce_t_all.sum() / denom
        kl = kl_all.sum() / denom
    else:
        ce_s = ce_s_all.sum()
        ce_t = ce_t_all.sum()
        kl = kl_all.sum()
    return ce_s + ce_t + kd_weight * kl, ce_s, ce_t, kl


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_torch_backend_reduction_contract_with_kl(reduction):
    hs, ht, weight, labels = _valid_cpu_inputs()
    labels = labels.clone()
    labels[1] = -100
    kd_weight = 0.35
    temperature = 1.7

    out = orda_ce_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_ce_kernel.TiedTeacher(ht),
        kd_weight=kd_weight,
        temperature=temperature,
        reduction=reduction,
        backend="torch",
    )
    expected = _manual_tied_reference(
        hs, ht, weight, labels,
        kd_weight=kd_weight, temperature=temperature, reduction=reduction,
    )

    assert torch.allclose(out.loss, expected[0])
    assert torch.allclose(out.student_ce, expected[1])
    assert torch.allclose(out.teacher_ce, expected[2])
    assert torch.allclose(out.kl, expected[3])


def test_reported_components_are_detached_but_loss_backprops():
    hs, ht, weight, labels = _valid_cpu_inputs()
    out = orda_ce_kernel.distillation_loss(
        hs,
        weight,
        labels,
        orda_ce_kernel.TiedTeacher(ht),
        kd_weight=0.25,
        backend="torch",
    )

    assert out.loss.requires_grad
    assert not out.student_ce.requires_grad
    assert not out.teacher_ce.requires_grad
    assert not out.kl.requires_grad
    out.loss.backward()
    assert hs.grad is not None
    assert ht.grad is not None
    assert weight.grad is not None


def test_explicit_triton_backend_fails_fast_on_cpu():
    hs, ht, weight, labels = _valid_cpu_inputs()
    with pytest.raises(RuntimeError, match="backend='triton'"):
        orda_ce_kernel.distillation_loss(
            hs,
            weight,
            labels,
            orda_ce_kernel.TiedTeacher(ht),
            backend="triton",
        )


def test_module_wrapper_matches_functional_torch_backend():
    hs, ht, weight, labels = _valid_cpu_inputs()
    teacher = orda_ce_kernel.TiedTeacher(ht)
    module = orda_ce_kernel.DistillationLoss(
        kd_weight=0.25,
        temperature=1.3,
        backend="torch",
    )

    out_module = module(hs, weight, labels, teacher)
    out_function = orda_ce_kernel.distillation_loss(
        hs,
        weight,
        labels,
        teacher,
        kd_weight=0.25,
        temperature=1.3,
        backend="torch",
    )

    assert torch.allclose(out_module.loss, out_function.loss)
    assert torch.allclose(out_module.student_ce, out_function.student_ce)
    assert torch.allclose(out_module.teacher_ce, out_function.teacher_ce)
    assert torch.allclose(out_module.kl, out_function.kl)
