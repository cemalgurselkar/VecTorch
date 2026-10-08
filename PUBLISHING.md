# Publishing Vectorch

This checklist is for maintainers. Building a package is safe and local;
publishing allocates a permanent version on a package index and requires the
maintainer's credentials or a configured Trusted Publisher.

## Before the first release

1. Confirm that the `vectorch` project name is still available on PyPI.
2. Create the PyPI project through its first upload, or configure a pending
   Trusted Publisher for the GitHub repository.
3. Ensure the release commit has a clean worktree and CI is green.

## Validate the release

Update the version in `pyproject.toml`, then refresh the lock file:

```bash
uv lock
```

Run all local checks:

```bash
uv run pytest
uv run mypy src tests benchmarks
uv run ruff check .
uv build --no-sources
```

The build must produce both files below, with the selected version substituted:

```text
dist/vectorch-0.1.0-py3-none-any.whl
dist/vectorch-0.1.0.tar.gz
```

Install the wheel into an isolated environment and run the example before
uploading it:

```bash
uv venv /tmp/vectorch-release-check
uv pip install --python /tmp/vectorch-release-check/bin/python \
  dist/vectorch-0.1.0-py3-none-any.whl
/tmp/vectorch-release-check/bin/python examples/quickstart.py
```

## Upload

For a manual token-based upload:

```bash
export UV_PUBLISH_TOKEN='<PyPI API token>'
uv publish
```

Do not commit a token or place it directly in shell history. With PyPI Trusted
Publishing, configure the GitHub repository, workflow name, and environment in
PyPI and let the release workflow request a short-lived identity token.

After upload, test the registry artifact independently of this checkout:

```bash
uv run --with vectorch --no-project -- \
  python -c "from vectorch import Vectorch; print(Vectorch)"
```

PyPI distributions are immutable: if a release is wrong, publish a new version
instead of trying to overwrite the existing files.
