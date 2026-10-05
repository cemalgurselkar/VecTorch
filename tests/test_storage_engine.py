import numpy as np

from vectorch.storage.engine import StorageEngine


def test_add_and_resolve_record():
    storage = StorageEngine(dimension=3)

    internal_id = storage.add(
        external_id="doc-1",
        vector=np.array([1, 2, 3], dtype=np.float32),
        metadata={"title": "hello"},
    )

    assert internal_id == 0
    assert storage.get_external_id(0) == "doc-1"
    assert storage.get_internal_id("doc-1") == 0
    assert storage.get_metadata(0) == {"title": "hello"}


def test_delete_marks_record():
    storage = StorageEngine(dimension=3)

    storage.add(
        "doc-1",
        np.array([1, 2, 3], dtype=np.float32),
    )

    storage.delete("doc-1")

    assert storage.is_deleted(0)