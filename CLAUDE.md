# spikedist

Pure-Python spike-train distance metrics. Zero runtime dependencies.

## Commands

- Create env and install: `uv venv && uv pip install -e ".[dev]"`
- Test: `uv run pytest -q`
- Lint: `uv run ruff check .` (format with `uv run ruff format .`)
- Types: `uv run mypy src`
- Build: `uv build` (then `uv run twine check dist/*` before publishing)

## Architecture

`src/spikedist/` with one module per metric:
- `_validate.py` shared input validation; returns sorted finite spike times
- `victor_purpura.py` edit distance via an O(n*m) dynamic program
- `van_rossum.py` kernel distance via the closed form of exponential-kernel inner products
- `__init__.py` public surface

See `docs/architecture.md` for the algorithms and the van Rossum normalization.

## Conventions

- Pure functions, strict typing, zero runtime dependencies.
- Metric parameters (`cost`, `tau`) are required keyword-only arguments; no default values.
- Spikes are an unordered set of times, sorted internally.
- Validate inputs and raise clear ValueError messages on bad data.

## Testing rules

- Exact closed-form reference values for each metric (see README definitions).
- Property tests (Hypothesis) for metric axioms: identity, symmetry, non-negativity, triangle inequality.
- Bug fixes start with a failing test.

## Release

- Semantic versioning; update `CHANGELOG.md` and `__version__`.
- Gates: `uv run pytest && uv run ruff check . && uv run mypy src && uv build && uv run twine check dist/*`.
- Prefer TestPyPI first, then PyPI. Tag `vX.Y.Z` and a GitHub release.

## Style

- No em dash characters in docs, comments, or commit messages.
- Comments explain non-obvious reasoning only.
