import numpy as np

from vectorch.storage.engine import StorageEngine


def test_storage_snapshot_round_trip():
    storage = StorageEngine(dimension=3)

    storage.add(
        "a",
        np.array([1, 2, 3], dtype=np.float32),
        {"kind": "first"},
    )
    storage.add(
        42,
        np.array([4, 5, 6], dtype=np.float32),
        None,
    )
    storage.add(
        "deleted",
        np.array([7, 8, 9], dtype=np.float32),
        {"kind": "deleted"},
    )

    storage.delete("deleted")

    snapshot = storage.snapshot()

    restored = StorageEngine.from_snapshot(
        dimension=3,
        snapshot=snapshot,
    )

    assert restored.total_count() == 3
    assert restored.count() == 2

    assert restored.get_internal_id("a") == 0
    assert restored.get_internal_id(42) == 1
    assert restored.get_internal_id("deleted") == 2

    np.testing.assert_array_equal(
        restored.get_vector(0),
        np.array([1, 2, 3], dtype=np.float32),
    )

    assert restored.get_metadata(0) == {
        "kind": "first"
    }
    assert restored.get_metadata(1) is None
    assert restored.is_deleted(2)