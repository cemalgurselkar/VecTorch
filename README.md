# Vectorch

Vectorch is an embedded, persistent vector database for Python. It runs in
your process, stores data on the local filesystem, and currently provides an
exact NumPy-backed flat index for cosine, dot-product, and squared-L2 search.

> **Alpha:** Vectorch `0.1.0` is an exact-search baseline, not an ANN engine.
> Expect search time to grow with the number and dimensionality of vectors.
> The on-disk format and public API may evolve before `1.0`.

## Features

- Embedded operation: no server or network service to manage.
- Persistent collections with atomic generation-based snapshots.
- Exact `flat` search with `cosine`, `dot`, and squared `l2` metrics.
- String or signed 64-bit integer IDs.
- JSON-compatible metadata.
- Logical deletion with active and total record counts.
- Context-manager lifecycle and explicit durability with `save()`.
- Reproducible benchmarks from 10K through 1M vectors.

## Installation

Vectorch requires Python 3.11 or newer.

Once the first release is published to PyPI:

```bash
pip install vectorch
```

or:

```bash
uv add vectorch
```

To install the current checkout instead:

```bash
git clone https://github.com/cemalgurselkar/VecTorch.git vectorch
cd vectorch
uv sync --dev
```

The PyPI commands above become available only after the maintainer uploads the
built release. See [Publishing](#publishing) for the final release step.

## Quick start

```python
from pathlib import Path

import numpy as np

from vectorch import Vectorch

database_path = Path("./vectorch-data")

with Vectorch(database_path) as db:
    documents = db.create_collection(
        "documents",
        dimension=3,
        metric="cosine",
    )
    documents.add(
        "python",
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        {"title": "Python"},
    )
    documents.add(
        "database",
        np.array([0.8, 0.2, 0.0], dtype=np.float32),
        {"title": "Databases"},
    )

    results = documents.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        k=2,
    )

    for result in results:
        print(result.id, result.score, result.metadata)
```

Exiting the context saves dirty collections and closes the database. Reopen
the same path and call `get_collection()` to restore persisted data:

```python
with Vectorch(database_path) as db:
    documents = db.get_collection("documents")
    print(documents.count())
```

## API overview

```python
db = Vectorch("./data")

collection = db.create_collection(
    name="items",
    dimension=384,
    metric="cosine",  # cosine | dot | l2
    index="flat",     # flat is the implemented index in 0.1.0
)

collection.add("item-1", vector, {"source": "example"})
results = collection.search(query, k=10)
stored_vector = collection.get("item-1")
collection.delete("item-1")

active_records = collection.count()
all_records_including_tombstones = collection.total_count()

collection.save()
db.drop_collection("items")
db.close()
```

For cosine and dot product, larger scores rank first. For `l2`, the score is
squared Euclidean distance and smaller scores rank first. Input vectors are
validated and stored internally as contiguous `float32` arrays.

## Durability model

Mutations update in-memory state first. They become durable when
`collection.save()`, `collection.close()`, or `db.close()` succeeds. The
context-manager form calls `close()` automatically.

Vectorch writes a complete new snapshot and then atomically publishes it as
the current generation. Version `0.1.0` does not include a write-ahead log or
per-mutation durability; changes made after the last successful save may be
lost if the process crashes.

## Docker

Docker is optional because Vectorch is a library, not a standalone server. The
included image installs the package in a clean Python environment and runs the
quick-start example:

```bash
docker build -t vectorch .
docker run --rm vectorch
```

Persist the example database on the host with:

```bash
mkdir -p vectorch-data
docker run --rm \
  -e VECTORCH_DB_PATH=/data/demo \
  -v "$(pwd)/vectorch-data:/data" \
  vectorch
```

## Benchmarks

The checked-in benchmark tools exercise the public API, validate exact results
against an independent NumPy oracle, capture environment metadata, and isolate
each scale in a fresh subprocess.

The current 128-dimensional cosine baseline (`k=10`, 1,000 measured queries,
metadata disabled) produced these results on an 8-logical-CPU Linux development
machine:

| Vectors | p50 | p99 | QPS |
|---:|---:|---:|---:|
| 10K | 1.406 ms | 5.269 ms | 519.97 |
| 25K | 4.097 ms | 10.082 ms | 202.47 |
| 50K | 8.956 ms | 15.816 ms | 103.51 |
| 100K | 19.969 ms | 27.615 ms | 48.59 |
| 250K | 59.913 ms | 117.062 ms | 15.59 |
| 500K | 111.326 ms | 150.390 ms | 8.75 |
| 1M | 199.701 ms | 285.300 ms | 4.88 |

These are machine-specific development results, not universal performance
claims. The worktree was dirty and BLAS/OpenMP thread counts were not pinned.
Read [report.md](report.md) for the complete diagnosis and limitations, and
[benchmarks/README.md](benchmarks/README.md) for reproducible commands.

Run a short smoke benchmark:

```bash
uv run python -m benchmarks.runner --profile quick
```

Run the isolated 10K-to-1M suite:

```bash
uv run python -m benchmarks.suite
```

The 1M suite requires several GiB of free memory and intentionally performs
full scans, so it is not a CI workload.

## Development

```bash
uv sync --dev
uv run pytest
uv run mypy src tests benchmarks
uv run ruff check .
uv build --no-sources
```

GitHub Actions runs tests, type checks, linting, and a package build on every
push and pull request. The million-vector benchmark remains an explicit local
operation.

## Publishing

The project builds both a source distribution and a platform-independent
wheel:

```bash
uv build --no-sources
```

Inspect and test the files in `dist/`, then publish with PyPI credentials or a
configured Trusted Publisher:

```bash
uv publish
```

Publishing is intentionally not performed by tests or by the Docker build.
Package names are allocated by PyPI at upload time, so confirm that `vectorch`
is still available immediately before the first release.

## Current limitations

- Only the exact `flat` index is implemented; HNSW and IVF are future work.
- Search is one query at a time and scans every stored vector.
- Cosine search currently recomputes corpus norms for every query. Profiling
  identifies norm caching as the first optimization target.
- No metadata filtering, updates/upserts, compaction, concurrent-writer
  protocol, or write-ahead log yet.
- Deleted IDs remain reserved and cannot currently be inserted again.

See [vectorch_remaining_roadmap.md](vectorch_remaining_roadmap.md) for the
development roadmap.

## License

Vectorch is distributed under the [MIT License](LICENSE).
