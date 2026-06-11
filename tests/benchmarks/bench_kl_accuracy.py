"""

Trong Triton, mặc dù code được viết dùng FP16, nhưng bản chất Triton tự động cast lên FP32 khi tính toán nên độ chính xác thực tế sẽ cao hơn là fp16_full. Vì vậy, biến thể fp16_f32_row_sum thực hiện 2 thay đổi chiến lược để đạt độ chính xác tương đương Triton:
  - Ép kiểu logits lên FP32 (.float()) ngay trước khi chia T, đảm bảo đạo hàm chảy ngược ở bước backward được tính toán hoàn toàn trên FP32.
  - Dùng reduction="none" để tính ma trận KL ở FP32, sau đó sum từng hàng, nhân T^2 để đạt mức độ chính xác tương tự Triton.

"""

from __future__ import annotations

import argparse
import gc

from tests.benchmarks.common import (
    add_output_args,
    benchmark_row,
    collect_metadata,
    require_cuda_kernel_or_skip,
    write_artifacts,
)
from tests.utils.env import parse_configs, set_seed
from tests.utils.reference import cosine_sim, max_abs_diff, mean_abs_diff
from tests.utils.timing import is_oom_exception


def main() -> None:
    parser = argparse.ArgumentParser(description="KL numerical accuracy benchmark")
    parser.add_argument("--configs", type=parse_configs, default=parse_configs("32x256,16x512,8x1024"))
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--kl-weight", type=float, default=0.4)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--lambda-student", type=float, default=1.0)
    parser.add_argument("--sample-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    add_output_args(parser)
    args = parser.parse_args()
    if not (0.0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0, 1]")

    runtime = require_cuda_kernel_or_skip("bench_kl_accuracy", args)
    torch = runtime.torch
    F = torch.nn.functional
    from orda_ce_kernel.utils.dispatcher import dynamic_chunk

    def fp32_full(hs_f32, ht_f32, w_f32, target):
        hs = hs_f32.clone().requires_grad_(True)
        ht = ht_f32.clone().requires_grad_(True)
        w = w_f32.clone().requires_grad_(True)
        logits_s = F.linear(hs, w)
        logits_t = F.linear(ht, w)
        ce_s = F.cross_entropy(logits_s, target)
        ce_t = F.cross_entropy(logits_t, target)
        log_p_s = F.log_softmax(logits_s / args.kl_temperature, dim=-1)
        p_t = F.softmax((logits_t / args.kl_temperature).detach(), dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (args.kl_temperature ** 2)
        total = ce_t + args.lambda_student * ce_s + args.kl_weight * kl
        total.backward()
        return kl.detach(), total.detach(), hs.grad.detach().float()

    def fp16_full(hs_f32, ht_f32, w_f32, target):
        hs = hs_f32.half().clone().requires_grad_(True)
        ht = ht_f32.half().clone().requires_grad_(True)
        w = w_f32.half().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits_s = F.linear(hs, w)
            logits_t = F.linear(ht, w)
            ce_s = F.cross_entropy(logits_s.float(), target)
            ce_t = F.cross_entropy(logits_t.float(), target)
            log_p_s = F.log_softmax(logits_s / args.kl_temperature, dim=-1)
            p_t = F.softmax((logits_t / args.kl_temperature).detach(), dim=-1)
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean").float() * (args.kl_temperature ** 2)
            total = ce_t + args.lambda_student * ce_s + args.kl_weight * kl
        total.backward()
        return kl.detach(), total.detach(), hs.grad.detach().float()

    def fp16_f32_row_sum(hs_f32, ht_f32, w_f32, target):
        hs = hs_f32.half().clone().requires_grad_(True)
        ht = ht_f32.half().clone().requires_grad_(True)
        w = w_f32.half().clone().requires_grad_(True)
        logits_s = F.linear(hs, w)
        logits_t = F.linear(ht, w)
        ce_s = F.cross_entropy(logits_s.float(), target)
        ce_t = F.cross_entropy(logits_t.float(), target)

        log_p_s = F.log_softmax(logits_s.float() / args.kl_temperature, dim=-1)
        with torch.no_grad():
            p_t = F.softmax(logits_t.float() / args.kl_temperature, dim=-1)
        kl_rows = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
        kl = kl_rows.sum() * (args.kl_temperature ** 2) / max(target.numel(), 1)
        total = ce_t + args.lambda_student * ce_s + args.kl_weight * kl
        total.backward()
        return kl.detach(), total.detach(), hs.grad.detach().float()

    def sample_25(hs_f32, ht_f32, w_f32, target):
        hs = hs_f32.half().clone().requires_grad_(True)
        ht = ht_f32.half().clone().requires_grad_(True)
        w = w_f32.half().clone().requires_grad_(True)
        total_tokens = hs.shape[0]
        sample_size = max(int(total_tokens * args.sample_frac), 1)
        with torch.autocast("cuda", dtype=torch.float16):
            ce, *_ = dynamic_chunk(
                hs,
                ht,
                w,
                target,
                lambda_student=args.lambda_student,
                kl_weight=0.0,
                kl_temperature=args.kl_temperature,
            )
            idx = torch.randperm(total_tokens, device=hs.device)[:sample_size]
            log_p_s = F.log_softmax(F.linear(hs[idx], w) / args.kl_temperature, dim=-1)
            p_t = F.softmax((F.linear(ht[idx].detach(), w.detach()) / args.kl_temperature), dim=-1)
            kl = F.kl_div(log_p_s, p_t.detach(), reduction="batchmean").float() * (args.kl_temperature ** 2)
            total = ce + args.kl_weight * kl
        total.backward()
        return kl.detach(), total.detach(), hs.grad.detach().float()

    def triton_full(hs_f32, ht_f32, w_f32, target):
        hs = hs_f32.half().clone().requires_grad_(True)
        ht = ht_f32.half().clone().requires_grad_(True)
        w = w_f32.half().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            total, _, _, kl = dynamic_chunk(
                hs,
                ht,
                w,
                target,
                lambda_student=args.lambda_student,
                kl_weight=args.kl_weight,
                kl_temperature=args.kl_temperature,
            )
        total.backward()
        return kl.detach(), total.detach(), hs.grad.detach().float()

    print("=" * 104)
    print("KL accuracy benchmark")
    print("=" * 104)
    print(f"{'Config':<12} | {'Variant':<18} | {'KL loss':>12} | {'KL rel err %':>12} | {'grad cos':>12} | {'mean abs':>12} | {'max abs':>12}")
    print("-" * 104)
    rows: list[dict] = []

    for batch, seq in args.configs:
        set_seed(torch, args.seed)
        bt = batch * seq
        hs = torch.randn(bt, args.hidden_dim, device="cuda", dtype=torch.float32)
        ht = torch.randn(bt, args.hidden_dim, device="cuda", dtype=torch.float32)
        w = torch.randn(args.vocab_size, args.hidden_dim, device="cuda", dtype=torch.float32)
        target = torch.randint(0, args.vocab_size, (bt,), device="cuda")

        try:
            ref_kl, ref_total, ref_grad = fp32_full(hs, ht, w, target)
        except Exception as exc:
            if not is_oom_exception(torch, exc):
                raise
            print(f"{batch}x{seq:<7} | {'fp32_full':<18} | {'OOM':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12}")
            rows.append(
                benchmark_row(
                    config=f"{batch}x{seq}",
                    method="fp32_full",
                    batch=batch,
                    seq=seq,
                    bt=bt,
                    vocab_size=args.vocab_size,
                    hidden_dim=args.hidden_dim,
                    status="oom",
                )
            )
            continue

        print(
            f"{batch}x{seq:<7} | {'fp32_full':<18} | {ref_kl.item():>12.6f} | "
            f"{0.0:>12.4f} | {1.0:>12.8f} | {0.0:>12.4e} | {0.0:>12.4e}"
        )
        rows.append(
            benchmark_row(
                config=f"{batch}x{seq}",
                method="fp32_full",
                batch=batch,
                seq=seq,
                bt=bt,
                vocab_size=args.vocab_size,
                hidden_dim=args.hidden_dim,
                kl_loss=float(ref_kl.item()),
                ref_kl_loss=float(ref_kl.item()),
                total_loss=float(ref_total.item()),
                kl_rel_err_pct=0.0,
                grad_cosine=1.0,
                mean_abs_diff=0.0,
                max_abs_diff=0.0,
                status="ok",
            )
        )

        variants = [
            ("fp16_full", fp16_full),
            ("fp16_f32_row_sum", fp16_f32_row_sum),
            ("sample_25", sample_25),
            ("triton", triton_full),
        ]
        for name, runner in variants:
            try:
                kl, total, grad = runner(hs, ht, w, target)
                rel = abs(float(kl.item() - ref_kl.item())) / max(abs(float(ref_kl.item())), 1e-12) * 100.0
                grad_cos = cosine_sim(torch, ref_grad, grad)
                mean_abs = mean_abs_diff(ref_grad, grad)
                max_abs = max_abs_diff(ref_grad, grad)
                print(
                    f"{batch}x{seq:<7} | {name:<18} | {kl.item():>12.6f} | {rel:>12.4f} | "
                    f"{grad_cos:>12.8f} | {mean_abs:>12.4e} | {max_abs:>12.4e}"
                )
                rows.append(
                    benchmark_row(
                        config=f"{batch}x{seq}",
                        method=name,
                        batch=batch,
                        seq=seq,
                        bt=bt,
                        vocab_size=args.vocab_size,
                        hidden_dim=args.hidden_dim,
                        kl_loss=float(kl.item()),
                        ref_kl_loss=float(ref_kl.item()),
                        total_loss=float(total.item()),
                        kl_rel_err_pct=rel,
                        grad_cosine=grad_cos,
                        mean_abs_diff=mean_abs,
                        max_abs_diff=max_abs,
                        sample_frac=args.sample_frac if name == "sample_25" else None,
                        status="ok",
                    )
                )
            except Exception as exc:
                if not is_oom_exception(torch, exc):
                    raise
                print(f"{batch}x{seq:<7} | {name:<18} | {'OOM':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12}")
                rows.append(
                    benchmark_row(
                        config=f"{batch}x{seq}",
                        method=name,
                        batch=batch,
                        seq=seq,
                        bt=bt,
                        vocab_size=args.vocab_size,
                        hidden_dim=args.hidden_dim,
                        status="oom",
                    )
                )
        gc.collect()
        torch.cuda.empty_cache()

    print("=" * 104)
    write_artifacts(
        rows,
        collect_metadata(torch, args, "bench_kl_accuracy"),
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()


