from __future__ import annotations

from functools import cache

from ._api_types import ChunkSize, EntropyGate, KernelConfig, KLVariant, Profile
from ._execution.chunking import (
    DEFAULT_MAX_FUSED_SIZE,
    normalize_chunk_request,
    normalize_max_fused_size,
)


# ── Profile resolution ───────────────────────────────────────────────────────
# Profiles are immutable frozen KernelConfigs; there are only three values, so
# the resolved object is cached and shared across calls.
@cache
def _resolve_profile(profile: Profile) -> KernelConfig:
    if profile == "balanced":
        return KernelConfig(
            grad_weight_accumulation="auto",
            max_fused_size=16384,
            autotune=True,
            refresh_autotune=False,
        )
    if profile == "fast":
        return KernelConfig(
            grad_weight_accumulation="auto",
            chunk_size=ChunkSize.fast(),
            max_fused_size=32768,
            autotune="Max",
            refresh_autotune=False,
        )
    if profile == "debug":
        return KernelConfig(
            grad_weight_accumulation="fp32",
            max_fused_size=DEFAULT_MAX_FUSED_SIZE,
            autotune=False,
            refresh_autotune=False,
        )
    raise ValueError(f"profile must be 'balanced', 'fast', or 'debug', got {profile!r}")


# ── Config merge ─────────────────────────────────────────────────────────────
def _resolve_config(profile: Profile, config: KernelConfig | None) -> KernelConfig:
    profile_config = _resolve_profile(profile)
    if config is None:
        config = KernelConfig()
    requested_chunk_size = config.chunk_size
    if requested_chunk_size is None:
        requested_chunk_size = profile_config.chunk_size
    if requested_chunk_size is None:
        requested_chunk_size = ChunkSize.dynamic()
    chunk_size, num_chunks, max_chunks = normalize_chunk_request(
        requested_chunk_size,
        config.num_chunks,
        config.max_chunks,
    )
    requested_max_fused_size = (
        profile_config.max_fused_size
        if config.max_fused_size is None
        else config.max_fused_size
    )
    if requested_max_fused_size is None:
        raise RuntimeError("profile resolution did not provide max_fused_size")
    max_fused_size = normalize_max_fused_size(requested_max_fused_size)
    refresh_autotune = (
        profile_config.refresh_autotune
        if config.refresh_autotune is None
        else config.refresh_autotune
    )
    autotune = profile_config.autotune if config.autotune is None else config.autotune
    if autotune is False and refresh_autotune:
        raise ValueError("refresh_autotune=True requires autotune=True or 'Max'")
    return KernelConfig(
        grad_weight_accumulation=(
            profile_config.grad_weight_accumulation
            if config.grad_weight_accumulation is None
            else config.grad_weight_accumulation
        ),
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        max_chunks=max_chunks,
        max_fused_size=max_fused_size,
        kl_weight=profile_config.kl_weight
        if config.kl_weight is None
        else config.kl_weight,
        kl_temperature=(
            profile_config.kl_temperature
            if config.kl_temperature is None
            else config.kl_temperature
        ),
        kl_teacher_temperature=(
            profile_config.kl_teacher_temperature
            if config.kl_teacher_temperature is None
            else config.kl_teacher_temperature
        ),
        kl_divergence=(
            profile_config.kl_divergence
            if config.kl_divergence is None
            else config.kl_divergence
        ),
        kl_alpha=config.kl_alpha,
        kl_beta=config.kl_beta,
        entropy_gate=config.entropy_gate,
        autotune=autotune,
        refresh_autotune=refresh_autotune,
        assume_valid_labels=(
            False if config.assume_valid_labels is None else config.assume_valid_labels
        ),
    )


# ── KL overrides ─────────────────────────────────────────────────────────────
def _resolve_kl_config(
    config: KernelConfig,
    kl_weight: float,
    kl_temperature: float,
    kl_divergence: KLVariant,
    kl_alpha: float | None,
    kl_beta: float | None,
    entropy_gate: EntropyGate | None,
    kl_teacher_temperature: float | None = None,
) -> tuple[
    float,
    float,
    float,
    str,
    float | None,
    float | None,
    EntropyGate | None,
]:
    """Resolve explicit config overrides for the KL parameters."""
    effective_kl_weight = kl_weight if config.kl_weight is None else config.kl_weight
    effective_kl_temperature = (
        kl_temperature if config.kl_temperature is None else config.kl_temperature
    )
    requested_teacher_temperature = (
        kl_teacher_temperature
        if config.kl_teacher_temperature is None
        else config.kl_teacher_temperature
    )
    effective_kl_teacher_temperature = (
        effective_kl_temperature
        if requested_teacher_temperature is None
        else requested_teacher_temperature
    )
    effective_kl_divergence = (
        kl_divergence if config.kl_divergence is None else str(config.kl_divergence)
    )
    effective_kl_alpha = kl_alpha if config.kl_alpha is None else config.kl_alpha
    effective_kl_beta = kl_beta if config.kl_beta is None else config.kl_beta
    effective_entropy_gate = (
        entropy_gate if config.entropy_gate is None else config.entropy_gate
    )
    return (
        effective_kl_weight,
        effective_kl_temperature,
        effective_kl_teacher_temperature,
        effective_kl_divergence,
        effective_kl_alpha,
        effective_kl_beta,
        effective_entropy_gate,
    )


# ── Teacher defaults ─────────────────────────────────────────────────────────
def _resolve_teacher_ce_weight(mode: str, teacher_ce_weight: float | None) -> float:
    if teacher_ce_weight is not None:
        return teacher_ce_weight
    return 1.0 if mode == "tied" else 0.0
