# Contributing to spikedist

Thanks for your interest. This project values correctness, a small surface area,
and zero runtime dependencies.

## Development

```sh
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run ruff check .
uv run mypy src
```

If you do not use uv, a standard virtual environment with `pip install -e ".[dev]"`
works the same way.

## Guidelines

- No runtime dependencies. NumPy and friends may be optional accelerators only.
- Every metric needs exact reference-value tests and property tests for the
  metric axioms it satisfies (identity, symmetry, non-negativity, and the
  triangle inequality where it holds).
- Metric parameters are required keyword-only arguments. No default values.
- Keep functions pure and fully typed; `mypy --strict` must pass.
- A bug fix starts with a failing test.
- Commit messages follow `type(scope): description`.

## Adding a metric

State the definition and its normalization in the docstring and the README, give
at least one exact closed-form test case, and add the relevant property tests.

## Reporting issues

Open an issue with the two spike trains, the parameters, the expected value with
a source, and the value you observed.
