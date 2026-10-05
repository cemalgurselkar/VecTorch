import numpy as np

from vectorch import Vectorch
import pytest

def test_end_to_end(tmp_path):
    db = Vectorch(tmp_path)

    collection = db.create_collection(
        name="documents",
        dimension=3,
        metric="cosine",
        index="flat",
    )

    collection.add(
        "x",
        np.array([1, 0, 0], dtype=np.float32),
    )

    collection.add(
        "y",
        np.array([0, 1, 0], dtype=np.float32),
    )

    results = collection.search(
        np.array([1, 0, 0], dtype=np.float32),
        k=1,
    )

    assert len(results) == 1
    assert results[0].id == "x"

def test_deleted_vector_is_not_returned(tmp_path):
    db = Vectorch(tmp_path)

    collection = db.create_collection(
        name="documents",
        dimension=2,
        metric="cosine",
        index="flat",
    )

    collection.add(
        "best",
        np.array([1.0, 0.0], dtype=np.float32),
    )
    collection.add(
        "second",
        np.array([0.9, 0.1], dtype=np.float32),
    )

    collection.delete("best")

    results = collection.search(
        np.array([1.0, 0.0], dtype=np.float32),
        k=1,
    )

    assert len(results) == 1
    assert results[0].id == "second"


def test_get_deleted_vector_fails(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    collection.add(
        "x",
        np.array([1.0, 0.0], dtype=np.float32),
    )

    collection.delete("x")

    with pytest.raises(KeyError):
        collection.get("x")

def test_duplicate_id_fails(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    vector = np.array([1.0, 0.0], dtype=np.float32)

    collection.add("x", vector)

    with pytest.raises(ValueError):
        collection.add("x", vector)


def test_search_k_larger_than_collection(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    collection.add(
        "x",
        np.array([1.0, 0.0], dtype=np.float32),
    )

    results = collection.search(
        np.array([1.0, 0.0], dtype=np.float32),
        k=100,
    )

    assert len(results) == 1


def test_search_empty_collection(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    results = collection.search(
        np.array([1.0, 0.0], dtype=np.float32),
        k=5,
    )

    assert results == []


def test_invalid_k_fails(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    collection.add(
        "x",
        np.array([1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError):
        collection.search(
            np.array([1.0, 0.0], dtype=np.float32),
            k=0,
        )


def test_wrong_query_dimension_fails(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=3)

    collection.add(
        "x",
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError):
        collection.search(
            np.array([1.0, 0.0], dtype=np.float32),
            k=1,
        )