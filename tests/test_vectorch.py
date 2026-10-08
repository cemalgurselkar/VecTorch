import numpy as np
import pytest

from vectorch import Vectorch


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

def test_get_does_not_expose_internal_vector(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=3)

    collection.add(
        "a",
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    vector = collection.get("a")
    vector[0] = 999.0

    stored = collection.get("a")

    assert stored[0] == 1.0

def test_count_excludes_deleted_vectors(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    collection.add("a", [1.0, 0.0])
    collection.add("b", [0.0, 1.0])

    collection.delete("a")

    assert collection.count() == 1
    assert collection.total_count() == 2


def test_delete_is_idempotent(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    collection.add("a", [1.0, 0.0])

    collection.delete("a")
    collection.delete("a")

    assert collection.count() == 0
    assert collection.total_count() == 1


def test_delete_unknown_id_fails(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    with pytest.raises(KeyError):
        collection.delete("missing")


def test_deleted_id_cannot_be_added_again(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    collection.add("a", [1.0, 0.0])
    collection.delete("a")

    with pytest.raises(ValueError):
        collection.add("a", [0.0, 1.0])

def test_vectorch_close_is_idempotent(tmp_path):
    db = Vectorch(tmp_path)

    db.close()
    db.close()


def test_vectorch_close_closes_collections(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    db.close()

    with pytest.raises(RuntimeError):
        collection.count()


def test_closed_vectorch_rejects_operations(tmp_path):
    db = Vectorch(tmp_path)
    db.close()

    with pytest.raises(RuntimeError):
        db.list_collections()

    with pytest.raises(RuntimeError):
        db.create_collection("test", dimension=2)


def test_drop_closes_collection(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    db.drop_collection("test")

    with pytest.raises(RuntimeError):
        collection.count()


def test_context_manager_closes_database(tmp_path):
    with Vectorch(tmp_path) as db:
        collection = db.create_collection("test", dimension=2)
        collection.add("a", [1.0, 0.0])

    with pytest.raises(RuntimeError):
        collection.count()

def test_search_rejects_boolean_k(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    collection.add("a", [1.0, 0.0])

    with pytest.raises((TypeError, ValueError)):
        collection.search([1.0, 0.0], k=True)


def test_cosine_zero_vector_score_is_zero(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection(
        "test",
        dimension=2,
        metric="cosine",
    )

    collection.add("zero", [0.0, 0.0])

    results = collection.search([1.0, 0.0], k=1)

    assert len(results) == 1
    assert results[0].id == "zero"
    assert results[0].score == pytest.approx(0.0)


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", "a/b", "a\\b"],
)
def test_rejects_collection_names_unsafe_for_filesystem(tmp_path, name):
    db = Vectorch(tmp_path)

    with pytest.raises(ValueError):
        db.create_collection(name, dimension=2)


@pytest.mark.parametrize("external_id", [True, False, 2**63, -(2**63) - 1])
def test_rejects_external_ids_not_supported_by_disk_format(
    tmp_path,
    external_id,
):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    with pytest.raises((TypeError, ValueError)):
        collection.add(external_id, [1.0, 0.0])

    assert collection.count() == 0


def test_boolean_id_cannot_alias_an_integer_id(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)
    collection.add(1, [1.0, 0.0])

    with pytest.raises(TypeError):
        collection.get(True)

    with pytest.raises(TypeError):
        collection.delete(True)


@pytest.mark.parametrize(
    "metadata",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": object()},
        {1: "non-string-key"},
        {"tuple": (1, 2)},
    ],
)
def test_rejects_metadata_not_preserved_by_json_round_trip(
    tmp_path,
    metadata,
):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)

    with pytest.raises(ValueError, match="JSON-compatible"):
        collection.add("a", [1.0, 0.0], metadata)

    assert collection.count() == 0


def test_rejects_cyclic_metadata(tmp_path):
    db = Vectorch(tmp_path)
    collection = db.create_collection("test", dimension=2)
    metadata = {}
    metadata["self"] = metadata

    with pytest.raises(ValueError, match="JSON-compatible"):
        collection.add("a", [1.0, 0.0], metadata)

    assert collection.count() == 0
