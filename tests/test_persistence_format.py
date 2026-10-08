import numpy as np
import pytest

from vectorch.persistence.format import (
    CorruptDataError,
    read_deleted,
    read_ids,
    read_metadata,
    read_vectors,
    write_deleted,
    write_ids,
    write_metadata,
    write_vectors,
)


def test_vectors_round_trip(tmp_path):
    path = tmp_path / "vectors.bin"

    vectors = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    write_vectors(path, vectors)

    restored = read_vectors(
        path,
        count=2,
        dimension=3,
    )

    np.testing.assert_array_equal(restored, vectors)
    assert restored.dtype == np.float32
    assert restored.flags.c_contiguous


def test_vectors_reject_invalid_file_size(tmp_path):
    path = tmp_path / "vectors.bin"
    path.write_bytes(b"invalid")

    with pytest.raises(CorruptDataError):
        read_vectors(
            path,
            count=2,
            dimension=3,
        )


def test_ids_round_trip(tmp_path):
    path = tmp_path / "ids.bin"

    ids = [
        "doc-1",
        42,
        "türkçe-id",
        -10,
    ]

    write_ids(path, ids)

    assert read_ids(path, count=4) == ids


def test_ids_reject_boolean(tmp_path):
    path = tmp_path / "ids.bin"

    with pytest.raises(TypeError):
        write_ids(path, [True])


def test_ids_reject_int_outside_int64(tmp_path):
    path = tmp_path / "ids.bin"

    with pytest.raises(ValueError):
        write_ids(path, [2**63])


def test_metadata_round_trip(tmp_path):
    path = tmp_path / "metadata.bin"

    metadata = [
        {"title": "hello", "page": 10},
        None,
        {},
        {"nested": {"value": True}},
    ]

    write_metadata(path, metadata)

    assert read_metadata(path, count=4) == metadata


def test_metadata_rejects_nan(tmp_path):
    path = tmp_path / "metadata.bin"

    with pytest.raises(ValueError):
        write_metadata(
            path,
            [{"score": float("nan")}],
        )


def test_metadata_reader_rejects_non_standard_json_constants(tmp_path):
    path = tmp_path / "metadata.bin"
    payload = b'{"score":NaN}'
    path.write_bytes(len(payload).to_bytes(4, "little") + payload)

    with pytest.raises(CorruptDataError):
        read_metadata(path, count=1)


def test_deleted_round_trip(tmp_path):
    path = tmp_path / "deleted.bin"

    deleted = np.array(
        [False, True, False, True],
        dtype=np.bool_,
    )

    write_deleted(path, deleted)

    restored = read_deleted(path, count=4)

    np.testing.assert_array_equal(
        restored,
        deleted,
    )
    assert restored.dtype == np.bool_


def test_deleted_rejects_invalid_value(tmp_path):
    path = tmp_path / "deleted.bin"
    path.write_bytes(bytes([0, 2, 1]))

    with pytest.raises(CorruptDataError):
        read_deleted(path, count=3)
