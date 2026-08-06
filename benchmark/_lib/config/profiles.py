from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.device import device_vram_gb
from ..core.validation import validate_positive_int
from ..model import ModelConfig

SMALL_PROFILE_VRAM_GB = 16


# ── Profile schemas ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainProfile:
    name: str
    teacher_config: ModelConfig
    student_config: ModelConfig
    vocabs: tuple[int, ...]
    seqs: tuple[int, ...]
    batch: int
    warmup: int
    update_steps: int
    passes: int
    temperature: float


@dataclass(frozen=True)
class AccuracyProfile:
    vocab: int
    seq_batches: tuple[tuple[int, int], ...]
    temperature: float


@dataclass(frozen=True)
class VramProfile:
    name: str
    batch: int
    vocabs: tuple[int, ...]
    seqs: tuple[int, ...]
    dims: tuple[int, ...]
    unequal_dim_pairs: tuple[tuple[int, int], ...]
    warmup: int
    measure_iters: int
    temperature: float
    num_chunks: int


# ── Profile values ───────────────────────────────────────────────────────────
SMALL_PROFILE = TrainProfile(
    name="small",
    teacher_config=ModelConfig(
        dim=1024, q_heads=8, kv_heads=4, n_layers=8, ffn_dim=2816
    ),
    student_config=ModelConfig(
        dim=512, q_heads=4, kv_heads=2, n_layers=4, ffn_dim=1408
    ),
    vocabs=(32_768, 65_536),
    seqs=(512, 1024),
    batch=8,
    warmup=5,
    update_steps=50,
    passes=2,
    temperature=3.0,
)

LARGE_PROFILE = TrainProfile(
    name="large",
    teacher_config=ModelConfig(
        dim=2048, q_heads=16, kv_heads=8, n_layers=16, ffn_dim=5632
    ),
    student_config=ModelConfig(
        dim=1024, q_heads=8, kv_heads=4, n_layers=8, ffn_dim=2816
    ),
    vocabs=(32_768, 131_072),
    seqs=(1024, 2048),
    batch=8,
    warmup=5,
    update_steps=50,
    passes=2,
    temperature=3.0,
)

VRAM_LARGE = VramProfile(
    name="large",
    batch=8,
    vocabs=(16_384, 32_768, 65_536),
    seqs=(512, 1024, 2048),
    dims=(512, 1024),
    unequal_dim_pairs=((512, 1024),),
    warmup=1,
    measure_iters=1,
    temperature=3.0,
    num_chunks=32,
)

# Keep hidden dimensions and chunk count identical so shared rows remain
# directly comparable with the large profile.
VRAM_SMALL = VramProfile(
    name="small",
    batch=8,
    vocabs=(16_384, 32_768),
    seqs=(512, 1024),
    dims=(512, 1024),
    unequal_dim_pairs=((512, 1024),),
    warmup=1,
    measure_iters=1,
    temperature=3.0,
    num_chunks=32,
)


# ── Profile selection ────────────────────────────────────────────────────────
def _is_small(device) -> bool:
    vram_gb = device_vram_gb(device)
    return vram_gb is None or vram_gb < SMALL_PROFILE_VRAM_GB


def _with_batch(profile, batch: int | None):
    if batch is None:
        return profile
    validate_positive_int("batch", batch)
    return replace(profile, batch=batch)


def benchmark_profile(device, *, batch: int | None = None) -> TrainProfile:
    profile = SMALL_PROFILE if _is_small(device) else LARGE_PROFILE
    return _with_batch(profile, batch)


def accuracy_profile(profile: TrainProfile) -> AccuracyProfile:
    return AccuracyProfile(
        vocab=16_384 if profile.name == "large" else 8_192,
        seq_batches=((2_048, 8), (520, 17)),
        temperature=1.7,
    )


def vram_profile(device, *, batch: int | None = None) -> VramProfile:
    profile = VRAM_SMALL if _is_small(device) else VRAM_LARGE
    return _with_batch(profile, batch)
