# Contributing

Thank you for considering a contribution to Vectorch. The project is in an
early alpha phase, so discuss large API, storage-format, or index changes in an
issue before investing in an implementation.

## Development setup

Install Python 3.11 or newer and uv, then run:

```bash
uv sync --dev
uv run pytest
uv run mypy src tests benchmarks
uv run ruff check .
```

Keep changes focused and add tests for observable behavior. Persistence changes
must cover restart behavior and malformed data. Search changes must preserve
the independent-oracle correctness checks.

## Performance changes

Start with the quick workload during development:

```bash
uv run python -m benchmarks.runner --profile quick
```

Before and after reports must use the same dataset size, dimension, metric,
`k`, seed, thread settings, hardware, and warm/cold conditions. Do not commit
generated profiles or benchmark reports unless they are deliberately selected
as a reviewed baseline.

## Pull requests

- Explain the user-visible behavior and tradeoffs.
- Include tests and documentation when the public contract changes.
- Confirm that tests, mypy, Ruff, and the package build pass.
- Do not mix unrelated formatting or refactoring into the change.

By submitting a contribution, you agree that it may be distributed under the
project's MIT License.
