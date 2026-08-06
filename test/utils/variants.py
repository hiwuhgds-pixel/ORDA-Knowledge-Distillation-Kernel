from __future__ import annotations

ALL_KL_VARIANTS = (
    "forward",
    "jeffreys",
    "mutual",
    "cojeffreys",
    "reverse",
    "alpha_divergence",
    "skew_forward",
    "skew_js",
    "entropy_adaptive",
    "skew_reverse",
    "renyi",
    "renyi_reverse",
    "dkd",
    "abkd",
    "triangular",
)

COUPLED_KL_VARIANTS = ("mutual", "cojeffreys")
DETACHED_KL_VARIANTS = tuple(
    variant for variant in ALL_KL_VARIANTS if variant not in COUPLED_KL_VARIANTS
)


def variant_kwargs(variant: str) -> dict[str, object]:
    if variant == "alpha_divergence":
        return {"kl_alpha": 0.3}
    if variant in ("skew_forward", "skew_js", "skew_reverse"):
        return {"kl_beta": 0.3}
    if variant == "renyi":
        return {"kl_alpha": 1.5}
    if variant == "renyi_reverse":
        return {"kl_alpha": 0.3}
    if variant == "dkd":
        return {"kl_alpha": 1.0, "kl_beta": 4.0}
    if variant == "abkd":
        return {"kl_alpha": 0.7, "kl_beta": 0.9}
    if variant == "entropy_adaptive":
        from orda_kernel import EntropyGate

        return {"entropy_gate": EntropyGate()}
    return {}
