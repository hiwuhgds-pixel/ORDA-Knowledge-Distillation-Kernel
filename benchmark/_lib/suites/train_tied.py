from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine.train import BuiltStep, TrainContext, TrainWorkload, optimizer_step
from ..losses.orda_loss import orda_loss, tied_teacher
from ..losses.spec import LossSpec
from ..losses.torch_loss import torch_loss_terms
from ..model import Block, ModelConfig, RMSNorm, RotaryEmbedding, Transformer
from .specs import SpecialCase
from .train import orda_grad_scaler

RDROP_DROPOUT = 0.1
KANGAROO_EXIT_LAYERS = {8: 2}
KANGAROO_DEFAULT_EXIT_LAYER = 3


# ── Workload helpers ─────────────────────────────────────────────────────────
def kangaroo_exit_layer(config: ModelConfig) -> int:
    return KANGAROO_EXIT_LAYERS.get(config.n_layers, KANGAROO_DEFAULT_EXIT_LAYER)


class KangarooAdapter(nn.Module):
    def __init__(self, config: ModelConfig, *, seq: int):
        super().__init__()
        self.rope = RotaryEmbedding(config.dim // config.q_heads, seq)
        self.block = Block(
            config.dim,
            config.q_heads,
            config.kv_heads,
            config.ffn_dim,
            seq,
        )
        self.norm = RMSNorm(config.dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(self.block(hidden, self.rope))


def _causal_batch(*, vocab: int, batch: int, seq: int, device: torch.device):
    tokens = torch.randint(0, vocab, (batch, seq + 1), device=device)
    return tokens[:, :-1], tokens[:, 1:]


def _flatten(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.reshape(-1, hidden.shape[-1])


def _forward_until(model: Transformer, tokens: torch.Tensor, *, exit_layer: int):
    hidden = model.tok(tokens)
    early_hidden = None
    for layer_index, block in enumerate(model.blocks, start=1):
        hidden = block(hidden, model.rope)
        if layer_index == exit_layer:
            early_hidden = hidden
    if early_hidden is None:
        raise ValueError(f"exit_layer={exit_layer} must be in [1, {len(model.blocks)}]")
    return early_hidden, model.norm(hidden)


def _run_blocks(model: Transformer, hidden: torch.Tensor, *, dropout: float = 0.0):
    if dropout == 0.0:
        for block in model.blocks:
            hidden = block(hidden, model.rope)
        return hidden

    for block in model.blocks:
        attention = block.attn(block.n1(hidden), model.rope)
        hidden = hidden + F.dropout(attention, p=dropout, training=model.training)
        feed_forward = block.ff(block.n2(hidden))
        hidden = hidden + F.dropout(feed_forward, p=dropout, training=model.training)
    return hidden


# ── Model workload ───────────────────────────────────────────────────────────
@dataclass
class TiedTrainWorkload(TrainWorkload):
    case: SpecialCase
    teacher_config: ModelConfig

    @property
    def spec(self) -> LossSpec:
        return LossSpec(
            temperature=self.case.temperature,
            kl_divergence=self.case.kl_divergence,
            student_ce_weight=self.case.student_ce_weight,
            teacher_ce_weight=self.case.teacher_ce_weight,
            kl_weight=self.case.kl_weight,
            tied_mode=self.case.tied_mode,
        )

    def build(self, ctx: TrainContext) -> BuiltStep:
        tokens, labels = _causal_batch(
            vocab=ctx.vocab, batch=ctx.batch, seq=ctx.seq, device=ctx.device
        )
        head_weight, labels_flat, hidden_pair, parameters = self._model(
            ctx, tokens, labels
        )
        optimizer = ctx.sgd(parameters)
        scaler = ctx.grad_scaler()
        spec = self.spec

        if ctx.mode == "torch-compile":

            def raw_loss():
                student_hidden, teacher_hidden = hidden_pair()
                return torch_loss_terms(
                    F.linear(student_hidden, head_weight),
                    F.linear(teacher_hidden, head_weight),
                    labels_flat,
                    spec,
                )

            forward_loss = torch.compile(raw_loss, mode="default")
        elif ctx.mode == "orda":
            compiled_pair = torch.compile(hidden_pair, mode="default")
            grad_scaler = orda_grad_scaler(scaler, ctx.dtype)

            def forward_loss():
                student_hidden, teacher_hidden = compiled_pair()
                return orda_loss(
                    student_hidden,
                    head_weight,
                    labels_flat,
                    tied_teacher(teacher_hidden, spec.tied_mode),
                    spec,
                    profile="fast",
                    grad_scaler=grad_scaler,
                )
        else:
            raise ValueError(f"Unknown mode: {ctx.mode}")

        return BuiltStep(
            step=optimizer_step(
                optimizer,
                scaler,
                forward_loss,
                dtype=ctx.dtype,
                device=ctx.device,
                use_amp=ctx.use_amp,
            ),
            optimizer=optimizer,
            scaler=scaler,
        )

    def _model(
        self, ctx: TrainContext, tokens: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Callable, tuple[nn.Parameter, ...]]:
        config = self.teacher_config
        labels_flat = labels.reshape(-1)

        if self.case.id == "kangaroo":
            base = Transformer(ctx.vocab, config, seq=ctx.seq).to(device=ctx.device)
            adapter = KangarooAdapter(config, seq=ctx.seq).to(device=ctx.device)
            if not ctx.use_amp:
                base.to(dtype=ctx.dtype)
                adapter.to(dtype=ctx.dtype)
            base.eval()
            for parameter in base.parameters():
                parameter.requires_grad_(False)
            adapter.train()

            with torch.no_grad():
                early_hidden, final_hidden = _forward_until(
                    base, tokens, exit_layer=kangaroo_exit_layer(config)
                )
                early_hidden = early_hidden.detach()
                final_hidden = _flatten(final_hidden.detach()).contiguous()
                head_weight = base.head.weight.detach().clone()
            del base

            def hidden_pair():
                return _flatten(adapter(early_hidden)), final_hidden

            return head_weight, labels_flat, hidden_pair, tuple(adapter.parameters())

        if self.case.id == "recursive_transformer":
            model = Transformer(
                ctx.vocab,
                config.replace(n_layers=config.n_layers // 2),
                seq=ctx.seq,
            ).to(device=ctx.device)
            if not ctx.use_amp:
                model.to(dtype=ctx.dtype)
            model.train()

            def hidden_pair():
                hidden = model.tok(tokens)
                hidden = _run_blocks(model, hidden)
                student_hidden = model.norm(hidden)
                hidden = _run_blocks(model, hidden)
                return _flatten(student_hidden), _flatten(model.norm(hidden))

            return (
                model.head.weight,
                labels_flat,
                hidden_pair,
                tuple(model.parameters()),
            )

        if self.case.id == "r_drop":
            model = Transformer(ctx.vocab, config, seq=ctx.seq).to(device=ctx.device)
            if not ctx.use_amp:
                model.to(dtype=ctx.dtype)
            model.train()

            def hidden_pair():
                duplicated = torch.cat((tokens, tokens), dim=0)
                hidden = model.tok(duplicated)
                hidden = F.dropout(hidden, p=RDROP_DROPOUT, training=model.training)
                hidden = _run_blocks(model, hidden, dropout=RDROP_DROPOUT)
                hidden = model.norm(hidden)
                student_hidden, teacher_hidden = hidden.chunk(2, dim=0)
                return _flatten(student_hidden), _flatten(teacher_hidden)

            return (
                model.head.weight,
                labels_flat,
                hidden_pair,
                tuple(model.parameters()),
            )

        raise ValueError(f"Unknown tied train case: {self.case.id}")


# ── Banner configuration ─────────────────────────────────────────────────────
def tied_banner_configs(case: SpecialCase, config: ModelConfig) -> tuple[str, str]:
    """Keep behavioural workload knobs descriptive; ModelConfig does not apply them."""
    base = config.as_dict()
    if case.id == "kangaroo":
        return (
            str({**base, "exit_layer": kangaroo_exit_layer(config)}),
            str({**base, "n_layers": 1}),
        )
    if case.id == "recursive_transformer":
        shared = {**base, "n_layers": config.n_layers // 2, "num_loops": 2}
        return str({**shared, "output_loop": 2}), str({**shared, "output_loop": 1})
    if case.id == "r_drop":
        shared = str({**base, "dropout": RDROP_DROPOUT})
        return shared, shared
    raise ValueError(f"Unknown tied train case: {case.id}")
