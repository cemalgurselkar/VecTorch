from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def latency_summary(latencies_seconds: Sequence[float]) -> dict[str, float]:
    if not latencies_seconds:
        raise ValueError("At least one latency sample is required")

    samples = np.asarray(latencies_seconds, dtype=np.float64)
    return {
        "min_ms": float(np.min(samples) * 1_000),
        "mean_ms": float(np.mean(samples) * 1_000),
        "p50_ms": float(np.percentile(samples, 50) * 1_000),
        "p95_ms": float(np.percentile(samples, 95) * 1_000),
        "p99_ms": float(np.percentile(samples, 99) * 1_000),
        "p99_9_ms": float(np.percentile(samples, 99.9) * 1_000),
        "max_ms": float(np.max(samples) * 1_000),
    }


def exact_top_k(
    query: NDArray[np.float32],
    vectors: NDArray[np.float32],
    *,
    metric: str,
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    if metric == "l2":
        differences = vectors - query
        scores = np.sum(differences * differences, axis=1)
        order = np.argsort(scores, kind="stable")[:k]
    elif metric == "dot":
        scores = vectors @ query
        order = np.argsort(-scores, kind="stable")[:k]
    elif metric == "cosine":
        dot = vectors @ query
        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query)
        scores = dot / np.maximum(vector_norms * query_norm, 1e-6)
        order = np.argsort(-scores, kind="stable")[:k]
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    ids = order.astype(np.int64, copy=False)
    return ids, scores[ids].astype(np.float32, copy=False)


def recall_at_k(expected: Sequence[int], actual: Sequence[int]) -> float:
    if not expected:
        raise ValueError("Expected IDs cannot be empty")

    return len(set(expected).intersection(actual)) / len(expected)


def correctness_summary(
    expected_ids: Sequence[Sequence[int]],
    actual_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(expected_ids) != len(actual_ids):
        raise ValueError("Expected and actual query counts must match")
    if not expected_ids:
        raise ValueError("At least one query result is required")

    recalls = [
        recall_at_k(expected, actual)
        for expected, actual in zip(expected_ids, actual_ids, strict=True)
    ]
    exact_matches = sum(
        set(expected) == set(actual)
        for expected, actual in zip(expected_ids, actual_ids, strict=True)
    )

    return {
        "mean_recall_at_k": float(np.mean(recalls)),
        "minimum_recall_at_k": float(np.min(recalls)),
        "exact_set_match_rate": exact_matches / len(expected_ids),
        "query_count": len(expected_ids),
    }


def score_error_summary(
    expected_ids: Sequence[Sequence[int]],
    expected_scores: Sequence[Sequence[float]],
    actual_ids: Sequence[Sequence[int]],
    actual_scores: Sequence[Sequence[float]],
) -> dict[str, Any]:
    query_counts = {
        len(expected_ids),
        len(expected_scores),
        len(actual_ids),
        len(actual_scores),
    }
    if len(query_counts) != 1:
        raise ValueError("ID and score query counts must match")

    errors: list[float] = []
    queries = zip(
        expected_ids,
        expected_scores,
        actual_ids,
        actual_scores,
        strict=True,
    )
    for (
        expected_query_ids,
        expected_query_scores,
        actual_query_ids,
        actual_query_scores,
    ) in queries:
        expected = dict(zip(expected_query_ids, expected_query_scores, strict=True))
        for internal_id, score in zip(
            actual_query_ids,
            actual_query_scores,
            strict=True,
        ):
            if internal_id in expected:
                errors.append(abs(float(score) - float(expected[internal_id])))

    if not errors:
        return {
            "compared_score_count": 0,
            "mean_absolute_score_error": None,
            "maximum_absolute_score_error": None,
        }

    return {
        "compared_score_count": len(errors),
        "mean_absolute_score_error": float(np.mean(errors)),
        "maximum_absolute_score_error": float(np.max(errors)),
    }
