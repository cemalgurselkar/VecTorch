from pathlib import Path

import numpy as np
import pytest

from vectorch import Vectorch
from vectorch.persistence.format import CorruptDataError


def _snapshot_directories(root: Path, name: str) -> list[Path]:
    snapshots = root / "collections" / name / "snapshots"
    return sorted(path for path in snapshots.iterdir() if path.name.isdigit())


def test_close_and_reopen_restores_complete_collection_state(tmp_path):
    with Vectorch(tmp_path) as database:
        documents = database.create_collection(
            "documents",
            dimension=3,
            metric="dot",
            index="flat",
        )
        documents.add(
            "first",
            [1.0, 0.0, 0.0],
            {
                "source": "example",
                "attributes": {"page": 1, "public": True},
                "tags": ["one", None],
            },
        )
        documents.add(42, [0.0, 1.0, 0.0])
        documents.add("deleted", [0.0, 0.0, 1.0])
        documents.delete("deleted")

    with Vectorch(tmp_path) as database:
        assert database.list_collections() == ["documents"]

        documents = database.get_collection("documents")
        assert documents.dimension == 3
        assert documents.config.metric.value == "dot"
        assert documents.config.index_type.value == "flat"
        assert documents.count() == 2
        assert documents.total_count() == 3

        np.testing.assert_array_equal(
            documents.get("first"),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            documents.get(42),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        with pytest.raises(KeyError):
            documents.get("deleted")

        results = documents.search([1.0, 0.0, 0.0], k=10)
        assert [result.id for result in results] == ["first", 42]
        assert results[0].metadata == {
            "source": "example",
            "attributes": {"page": 1, "public": True},
            "tags": ["one", None],
        }


def test_explicit_save_is_a_durability_boundary(tmp_path):
    database = Vectorch(tmp_path)
    documents = database.create_collection("documents", dimension=2)
    documents.add("saved", [1.0, 0.0])
    documents.save()

    reopened = Vectorch(tmp_path)
    try:
        restored = reopened.get_collection("documents")
        np.testing.assert_array_equal(
            restored.get("saved"),
            np.array([1.0, 0.0], dtype=np.float32),
        )
    finally:
        reopened.close()
        database.close()


def test_empty_and_multiple_collections_survive_restart(tmp_path):
    with Vectorch(tmp_path) as database:
        database.create_collection("empty", dimension=2)
        other = database.create_collection("other", dimension=4, metric="l2")
        other.add("record", [1.0, 2.0, 3.0, 4.0])

    with Vectorch(tmp_path) as database:
        assert database.list_collections() == ["empty", "other"]
        assert database.get_collection("empty").count() == 0
        assert database.get_collection("other").config.metric.value == "l2"


def test_clean_close_does_not_create_another_generation(tmp_path):
    with Vectorch(tmp_path) as database:
        documents = database.create_collection("documents", dimension=2)
        documents.add("a", [1.0, 0.0])

    assert len(_snapshot_directories(tmp_path, "documents")) == 1

    with Vectorch(tmp_path):
        pass

    assert len(_snapshot_directories(tmp_path, "documents")) == 1


def test_second_dirty_save_publishes_another_generation(tmp_path):
    with Vectorch(tmp_path) as database:
        documents = database.create_collection("documents", dimension=2)
        documents.add("a", [1.0, 0.0])
        documents.save()
        documents.add("b", [0.0, 1.0])
        documents.save()

    assert len(_snapshot_directories(tmp_path, "documents")) == 2


def test_drop_removes_persisted_collection_across_restart(tmp_path):
    with Vectorch(tmp_path) as database:
        documents = database.create_collection("documents", dimension=2)
        documents.add("a", [1.0, 0.0])

    with Vectorch(tmp_path) as database:
        database.drop_collection("documents")

    with Vectorch(tmp_path) as database:
        assert database.list_collections() == []
        with pytest.raises(ValueError):
            database.get_collection("documents")


def test_save_failure_keeps_collection_dirty_and_open(tmp_path, monkeypatch):
    database = Vectorch(tmp_path)
    documents = database.create_collection("documents", dimension=2)
    documents.add("a", [1.0, 0.0])

    persistence = documents._persistence
    original_save = persistence.save_collection

    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(persistence, "save_collection", fail_save)

    with pytest.raises(OSError, match="simulated write failure"):
        documents.save()

    assert documents.count() == 1

    monkeypatch.setattr(persistence, "save_collection", original_save)
    documents.save()
    database.close()

    with Vectorch(tmp_path) as reopened:
        assert reopened.get_collection("documents").count() == 1


def test_missing_snapshot_file_is_reported_on_database_open(tmp_path):
    with Vectorch(tmp_path) as database:
        database.create_collection("documents", dimension=2)

    snapshot = _snapshot_directories(tmp_path, "documents")[0]
    (snapshot / "metadata.bin").unlink()

    with pytest.raises(CorruptDataError, match="missing metadata.bin"):
        Vectorch(tmp_path)


def test_invalid_manifest_version_is_reported_on_database_open(tmp_path):
    Vectorch(tmp_path).close()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"format": "vectorch", "format_version": 999}',
        encoding="utf-8",
    )

    with pytest.raises(CorruptDataError, match="format version"):
        Vectorch(tmp_path)
