import numpy as np
import pytest

from vectorch import Vectorch
from vectorch.storage.vectors import VectorStorage
from vectorch.types import IndexType, Metric


def test_append_and_get():
    store = VectorStorage(dimension=3)

    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    internal_id = store.append(vector)

    assert internal_id == 0
    np.testing.assert_array_equal(store.get(0), vector)


def test_grows_when_capacity_exceeded():
    store = VectorStorage(dimension=2, initial_capacity=2)

    for i in range(3):
        store.append(
            np.array([i, i], dtype=np.float32)
        )

    assert len(store) == 3
    assert store.capacity >= 3


def test_rejects_wrong_dimension():
    store = VectorStorage(dimension=3)

    with pytest.raises(ValueError):
        store.append(
            np.array([1.0, 2.0], dtype=np.float32)
        )

def test_create_collection_normalizes_string_config(tmp_path):
    db = Vectorch(tmp_path)

    collection = db.create_collection(
        "docs",
        dimension=128,
        metric="cosine",
        index="flat",
    )

    assert collection.config.metric is Metric.COSINE
    assert collection.config.index_type is IndexType.FLAT


def test_create_collection_accepts_enum_config(tmp_path):
    db = Vectorch(tmp_path)

    collection = db.create_collection(
        "docs",
        dimension=128,
        metric=Metric.L2,
        index=IndexType.FLAT,
    )

    assert collection.config.metric is Metric.L2
    assert collection.config.index_type is IndexType.FLAT

def test_rejects_invalid_metric_at_creation(tmp_path):
    db = Vectorch(tmp_path)

    with pytest.raises(ValueError, match="Invalid metric"):
        db.create_collection(
            "docs",
            dimension=128,
            metric="invalid",
        )

def test_rejects_invalid_index_at_creation(tmp_path):
    db = Vectorch(tmp_path)

    with pytest.raises(ValueError, match="Invalid index"):
        db.create_collection(
            "docs",
            dimension=128,
            index="invalid",
        )

def test_list_collections(tmp_path):
    db = Vectorch(tmp_path)

    db.create_collection("a", dimension=2)
    db.create_collection("b", dimension=2)

    assert db.list_collections() == ["a", "b"]