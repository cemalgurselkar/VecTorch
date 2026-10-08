from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import NDArray

from vectorch.validation import validate_external_id, validate_metadata

FORMAT_VERSION = 1
VECTOR_DTYPE = np.dtype("<f4")

_ID_INT = 0
_ID_STR = 1

_UINT32 = struct.Struct("<I")
_INT64 = struct.Struct("<q")


class PersistenceError(RuntimeError):
    pass


class CorruptDataError(PersistenceError):
    pass


def write_vectors(
    path: Path,
    vectors: NDArray[np.float32],
) -> None:
    data = np.asarray(vectors, dtype=VECTOR_DTYPE, order="C")

    with path.open("wb") as file:
        file.write(data.tobytes(order="C"))


def read_vectors(
    path: Path,
    *,
    count: int,
    dimension: int,
) -> NDArray[np.float32]:
    expected_bytes = count * dimension * VECTOR_DTYPE.itemsize

    if path.stat().st_size != expected_bytes:
        raise CorruptDataError(
            f"Invalid vectors.bin size: expected "
            f"{expected_bytes}, got {path.stat().st_size}"
        )

    data = np.fromfile(path, dtype=VECTOR_DTYPE)

    return data.reshape(count, dimension).astype(
        np.float32,
        copy=False,
    )


def write_ids(
    path: Path,
    ids: list[str | int],
) -> None:
    with path.open("wb") as file:
        for external_id in ids:
            external_id = validate_external_id(external_id)

            if isinstance(external_id, int):
                payload = _INT64.pack(external_id)

                file.write(bytes([_ID_INT]))
                file.write(payload)
                continue

            if isinstance(external_id, str):
                payload = external_id.encode("utf-8")

                if len(payload) > 0xFFFFFFFF:
                    raise ValueError("External ID is too large")

                file.write(bytes([_ID_STR]))
                file.write(_UINT32.pack(len(payload)))
                file.write(payload)
                continue

            raise TypeError("External ID must be str or int")


def read_ids(
    path: Path,
    *,
    count: int,
) -> list[str | int]:
    ids: list[str | int] = []

    with path.open("rb") as file:
        for _ in range(count):
            type_bytes = file.read(1)

            if len(type_bytes) != 1:
                raise CorruptDataError("Unexpected end of ids.bin")

            type_code = type_bytes[0]

            if type_code == _ID_INT:
                payload = _read_exact(file, _INT64.size)
                ids.append(_INT64.unpack(payload)[0])

            elif type_code == _ID_STR:
                length_bytes = _read_exact(file, _UINT32.size)
                length = _UINT32.unpack(length_bytes)[0]

                payload = _read_exact(file, length)

                try:
                    ids.append(payload.decode("utf-8"))
                except UnicodeDecodeError as exc:
                    raise CorruptDataError(
                        "Invalid UTF-8 in ids.bin"
                    ) from exc

            else:
                raise CorruptDataError(
                    f"Unknown ID type code: {type_code}"
                )

        if file.read(1):
            raise CorruptDataError(
                "ids.bin contains trailing data"
            )

    return ids


def write_metadata(
    path: Path,
    metadata: list[dict[str, Any] | None],
) -> None:
    with path.open("wb") as file:
        for value in metadata:
            value = validate_metadata(value)
            payload = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")

            if len(payload) > 0xFFFFFFFF:
                raise ValueError("Metadata record is too large")

            file.write(_UINT32.pack(len(payload)))
            file.write(payload)


def read_metadata(
    path: Path,
    *,
    count: int,
) -> list[dict[str, Any] | None]:
    result: list[dict[str, Any] | None] = []

    with path.open("rb") as file:
        for _ in range(count):
            length_bytes = _read_exact(file, _UINT32.size)
            length = _UINT32.unpack(length_bytes)[0]

            payload = _read_exact(file, length)

            try:
                value = json.loads(
                    payload.decode("utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeDecodeError, ValueError) as exc:
                raise CorruptDataError(
                    "Invalid metadata.bin record"
                ) from exc

            if value is not None and not isinstance(value, dict):
                raise CorruptDataError(
                    "Metadata record must be an object or null"
                )

            result.append(value)

        if file.read(1):
            raise CorruptDataError(
                "metadata.bin contains trailing data"
            )

    return result


def write_deleted(
    path: Path,
    deleted: NDArray[np.bool_],
) -> None:
    data = np.asarray(deleted, dtype=np.uint8)
    path.write_bytes(data.tobytes())


def read_deleted(
    path: Path,
    *,
    count: int,
) -> NDArray[np.bool_]:
    if path.stat().st_size != count:
        raise CorruptDataError(
            f"Invalid deleted.bin size: expected {count}, "
            f"got {path.stat().st_size}"
        )

    data = np.fromfile(path, dtype=np.uint8)

    if np.any((data != 0) & (data != 1)):
        raise CorruptDataError(
            "deleted.bin contains invalid values"
        )

    return data.astype(np.bool_)


def _read_exact(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)

    if len(data) != size:
        raise CorruptDataError("Unexpected end of binary file")

    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")
