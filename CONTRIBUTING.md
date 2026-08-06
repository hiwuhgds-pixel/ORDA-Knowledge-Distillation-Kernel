# Contributing

## Development setup

Use Python 3.10 or newer and install the development dependencies:

```bash
python -m pip install -e ".[ci]"
```

Before submitting a change, run:

```bash
python -m ruff format --check src test scripts .github/scripts benchmark
python -m ruff check src test scripts .github/scripts benchmark
python -m mypy
python -m pytest -m "not gpu" test/unit test/correctness/test_torch_backend_cpu.py test/nightly/test_autotune_multiprocess.py
```

Changes to a timed autotune-contract source must also run:

```bash
python scripts/update_contract_digest.py
```

The script exits with status 1 when it updates the stored digest; rerun it to confirm the digest is stable. CUDA/HIP or Triton changes require the GPU matrix described in `docs/gpu-validation.md`; CPU-only results must not be presented as hardware parity.

Keep public behavior documented, comments concise, and new tests focused on a behavioral contract rather than formatting or implementation trivia.
