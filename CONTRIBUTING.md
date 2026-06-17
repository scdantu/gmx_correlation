# Contributing to gmx_correlation

## Development setup

```bash
# Clone and enter the repo
git clone https://github.com/sarathdantu/gmx_correlation.git
cd gmx_correlation

# Python dependencies (includes test tools)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No GROMACS installation is needed to run the Python test suite.

## Running tests

```bash
pytest tests/ -v                        # fast suite (~2 s)
pytest tests/test_te_validation.py -v  # TE validation (KNN, ~1 s)
```

All 29 tests must pass before a PR is merged.

## Code style

- **C++**: follow the existing style (2-space indent, `{}` on same line for short blocks). No external formatter enforced yet.
- **Python**: PEP 8. Lines ≤ 100 chars. No type annotations required but welcome.
- **No new dependencies** without discussion — the C++ side intentionally keeps GROMACS as its only external dependency.

## Adding a new estimator

1. Add a header in `lib/` declaring `void newmethod_corrmatrix(const t_traj*, double*, ...)`.
2. Implement in `lib/newmethod.cpp` (and optionally `lib/newmethod_metal.mm` / `lib/newmethod_gpu.cu`).
3. Add an option flag in `src/gmx_correlation.cpp` (`registerOptions`) and dispatch in `writeOutput`.
4. Add to `CMakeLists.txt` source list.
5. Add Python reference implementation + tests in `tests/test_newmethod.py` comparing against an analytical baseline.

## Submitting a PR

- Branch from `main`, name it `feature/...` or `fix/...`.
- Keep commits focused; one logical change per commit.
- Update `CHANGELOG.md` under an `[Unreleased]` heading.
- All tests must pass in CI before review.

## Reporting issues

Open an issue on GitHub with:
- OS, GROMACS version, compiler version.
- Minimal reproduction command and input files (or synthetic data).
- Expected vs. actual output.
