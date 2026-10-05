import numpy as np
import pytest

from vectorch.storage.vectors import VectorStorage


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