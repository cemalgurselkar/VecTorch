# Vectorch Benchmark Suite

This directory contains the reproducible FlatIndex baseline used before any
performance optimization. It measures the public Vectorch API rather than a
private kernel in isolation.

No generated result is treated as a universal performance claim. Keep the
machine information, configuration, Git revision, and dirty-worktree flag in
the JSON report when publishing numbers.

## Workload profiles

`quick` is a short local smoke workload:

- 2,000 vectors
- 64 dimensions
- 50 measured queries
- 10 warmup queries
- cosine, k=10

`standard` is the initial local baseline:

- 1,000,000 vectors
- 128 dimensions
- 1,000 measured queries
- 25 oracle-validated queries
- 50 warmup queries
- cosine, k=10
- metadata disabled by default to isolate the numeric/index workload

Both profiles use seed 42. The quick profile includes representative JSON
metadata; the million-vector standard profile does not. Every parameter can be
overridden from the command line.

## Run a baseline

From the repository root:

```bash
uv run python -m benchmarks.runner --profile quick
uv run python -m benchmarks.runner --profile standard
```

The recommended optimization baseline is the isolated scaling suite:

```bash
uv run python -m benchmarks.suite
```

It runs these sizes in ascending order:

```text
10K → 25K → 50K → 100K → 250K → 500K → 1M
```

Every size runs in a fresh Python subprocess. When one scenario finishes, its
temporary database and process memory are released before the next process
starts. This isolates Python objects, NumPy buffers, allocator state, and RSS.
The operating system page cache is not forcibly cleared because doing so is a
privileged machine-wide operation and would make the benchmark intrusive.

Each scenario writes its own JSON report. The suite directory also contains
`suite.json`, which is updated after every completed stage and summarizes the
latency, QPS, ingestion, persistence, memory, and correctness curve. If a stage
fails, earlier reports remain available and the manifest records the failed
size.

Custom suite example:

```bash
uv run python -m benchmarks.suite \
  --sizes 10000 100000 1000000 \
  --query-count 500 \
  --oracle-query-count 25 \
  --metric cosine \
  --no-metadata
```

After an optimization, compare two completed scaling curves with:

```bash
uv run python -m benchmarks.compare_suite \
  benchmarks/results/suite-before/suite.json \
  benchmarks/results/suite-after/suite.json
```

Positive percentages mean improvement. Latency, persistence time, and memory
improve by decreasing; throughput and QPS improve by increasing.

Reports are written under `benchmarks/results/`. Generated reports are ignored
by Git so that reviewed baseline files can be added deliberately when needed.

Example custom workload:

```bash
uv run python -m benchmarks.runner \
  --profile standard \
  --dataset-size 50000 \
  --dimension 384 \
  --query-count 500 \
  --oracle-query-count 25 \
  --warmup-queries 50 \
  --metric cosine \
  --k 10 \
  --seed 42 \
  --output benchmarks/results/cosine-50k-384d.json
```

Use `--metadata` for a separate 1M RAG-style run that includes metadata, or
`--no-metadata` to isolate vector and index costs.

## Measurements

Each JSON report contains:

- dataset generation time;
- insert throughput;
- FlatIndex build time, explicitly zero because it has no build phase;
- snapshot save and database reopen time;
- database size on disk;
- first-query latency before warmup;
- warm search min, mean, p50, p95, p99, p99.9, max, and QPS;
- current RSS checkpoints and process peak RSS when the platform exposes them;
- Recall@K, exact top-k set-match rate, and score error against an independent
  NumPy oracle;
- Python, NumPy, OS, CPU, Git revision, seed, and complete workload config.
- process CPU time/utilization and common BLAS/OpenMP thread environment values.

The benchmark uses one query at a time through `Collection.search()`. It
therefore measures the current public API path, including ID and metadata
resolution. Oracle computation is timed separately and excluded from search
latency and QPS. At 1M scale, the standard profile validates a deterministic
25-query subset against the full-sort oracle while measuring 1,000 searches.

## Million-vector resource expectations

The raw standard vector matrix is approximately 488 MiB. During ingestion,
snapshotting, reopen, and oracle validation, multiple vector-sized buffers can
coexist. Use a machine with several GiB of free RAM; 8 GiB or more is the
practical minimum for this profile. Enabling metadata increases memory use
substantially because one Python dictionary is created per vector.

The standard profile intentionally performs 1,000 full FlatIndex scans. It is
expected to take materially longer than the quick profile. This is the target
baseline for later norm caching, allocation reduction, batching, and ANN work,
not a CI workload.

## Profile the search hot path

```bash
uv run python -m benchmarks.profile_search --profile quick
uv run python -m benchmarks.profile_search --profile standard
```

This writes a binary `.prof` file, a cumulative-time text summary, and a JSON
manifest containing the workload and environment under `benchmarks/profiles/`.
Setup, ingestion, persistence, and warmup are excluded from the captured
profile; only measured searches are profiled.

Inspect a binary profile with standard Python tools:

```bash
uv run python -m pstats benchmarks/profiles/search-TIMESTAMP.prof
```

## Compare two reports

```bash
uv run python -m benchmarks.compare \
  benchmarks/results/baseline.json \
  benchmarks/results/candidate.json
```

Positive `improvement` means better in the comparison output. The tool knows
that throughput should increase while latency, disk size, and memory should
decrease.

Compare reports only when dataset, dimension, metric, k, seed, thread settings,
hardware, and warm/cold conditions match. A dirty Git worktree should be noted
when preserving a baseline.

## Measurement limitations

- RSS readings are approximate and platform-dependent.
- Linux current RSS comes from `/proc`; peak RSS uses `getrusage` where
  available.
- The first query is reported separately but is not a full cold-page-cache
  benchmark.
- NumPy and BLAS may use implementation-specific internal threading.
- Set thread controls before launching the process when comparing runs, for
  example `OPENBLAS_NUM_THREADS=1` or `OMP_NUM_THREADS=1`; the report records
  common thread-control environment variables.
- Background load, CPU frequency scaling, and thermal throttling affect tail
  latency.
- Run the standard workload multiple times before drawing conclusions.
- This stage does not add Numba, SIMD, batch search, or ANN optimizations.
