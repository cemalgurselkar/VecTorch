import json

import numpy as np
import pytest

from vectorch.persistence.format import CorruptDataError
from vectorch.persistence.manager import PersistenceManager
from vectorch.storage.engine import StorageEngine
from vectorch.types import CollectionConfig, Metric


def test_initializes_database_layout(tmp_path):
    PersistenceManager(tmp_path)

    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "collections").is_dir()

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest == {
        "format": "vectorch",
        "format_version": 1,
    }


def test_save_and_load_collection(tmp_path):
    manager = PersistenceManager(tmp_path)

    config = CollectionConfig(
        name="documents",
        dimension=3,
        metric=Metric.COSINE,
    )

    storage = StorageEngine(dimension=3)

    storage.add(
        "first",
        np.array([1, 2, 3], dtype=np.float32),
        {"title": "one"},
    )
    storage.add(
        42,
        np.array([4, 5, 6], dtype=np.float32),
        None,
    )
    storage.add(
        "deleted",
        np.array([7, 8, 9], dtype=np.float32),
        {"title": "gone"},
    )
    storage.delete("deleted")

    manager.save_collection(
        config,
        storage.snapshot(),
    )

    loaded_config, snapshot = (
        manager.load_collection("documents")
    )

    assert loaded_config == config

    restored = StorageEngine.from_snapshot(
        dimension=loaded_config.dimension,
        snapshot=snapshot,
    )

    assert restored.total_count() == 3
    assert restored.count() == 2
    assert restored.get_internal_id("first") == 0
    assert restored.get_internal_id(42) == 1
    assert restored.get_internal_id("deleted") == 2

    assert restored.get_metadata(0) == {
        "title": "one"
    }
    assert restored.get_metadata(1) is None
    assert restored.is_deleted(2)


def test_second_save_publishes_new_generation(tmp_path):
    manager = PersistenceManager(tmp_path)

    config = CollectionConfig(
        name="documents",
        dimension=2,
    )

    storage = StorageEngine(dimension=2)

    storage.add(
        "a",
        np.array([1, 2], dtype=np.float32),
        None,
    )

    manager.save_collection(
        config,
        storage.snapshot(),
    )

    current_path = (
        tmp_path
        / "collections"
        / "documents"
        / "CURRENT"
    )

    assert current_path.read_text().strip() == (
        "0000000000000001"
    )

    storage.add(
        "b",
        np.array([3, 4], dtype=np.float32),
        None,
    )

    manager.save_collection(
        config,
        storage.snapshot(),
    )

    assert current_path.read_text().strip() == (
        "0000000000000002"
    )

    _, snapshot = manager.load_collection(
        "documents"
    )

    assert len(snapshot.ids) == 2


def test_lists_persisted_collections(tmp_path):
    manager = PersistenceManager(tmp_path)

    for name in ("b", "a"):
        config = CollectionConfig(
            name=name,
            dimension=2,
        )
        storage = StorageEngine(dimension=2)

        manager.save_collection(
            config,
            storage.snapshot(),
        )

    assert manager.list_collections() == [
        "a",
        "b",
    ]


def test_drop_collection_removes_it(tmp_path):
    manager = PersistenceManager(tmp_path)

    config = CollectionConfig(
        name="documents",
        dimension=2,
    )
    storage = StorageEngine(dimension=2)

    manager.save_collection(
        config,
        storage.snapshot(),
    )

    manager.drop_collection("documents")

    assert manager.list_collections() == []

    with pytest.raises(KeyError):
        manager.load_collection("documents")


def test_rejects_corrupt_current(tmp_path):
    manager = PersistenceManager(tmp_path)

    config = CollectionConfig(
        name="documents",
        dimension=2,
    )
    storage = StorageEngine(dimension=2)

    manager.save_collection(
        config,
        storage.snapshot(),
    )

    current = (
        tmp_path
        / "collections"
        / "documents"
        / "CURRENT"
    )
    current.write_text(
        "garbage",
        encoding="ascii",
    )

    with pytest.raises(CorruptDataError):
        manager.load_collection("documents")


def test_rejects_corrupt_vectors(tmp_path):
    manager = PersistenceManager(tmp_path)

    config = CollectionConfig(
        name="documents",
        dimension=2,
    )
    storage = StorageEngine(dimension=2)

    storage.add(
        "a",
        np.array([1, 2], dtype=np.float32),
        None,
    )

    manager.save_collection(
        config,
        storage.snapshot(),
    )

    vectors = (
        tmp_path
        / "collections"
        / "documents"
        / "snapshots"
        / "0000000000000001"
        / "vectors.bin"
    )

    vectors.write_bytes(b"broken")

    with pytest.raises(CorruptDataError):
        manager.load_collection("documents")