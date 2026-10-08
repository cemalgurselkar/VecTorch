from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from vectorch.storage.engine import StorageSnapshot
from vectorch.types import CollectionConfig, IndexType, Metric
from vectorch.validation import validate_collection_name

from .format import (
    FORMAT_VERSION,
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

_MANIFEST_NAME = "manifest.json"
_COLLECTIONS_DIR = "collections"
_CURRENT_NAME = "CURRENT"
_SNAPSHOTS_DIR = "snapshots"

_CONFIG_NAME = "config.json"
_VECTORS_NAME = "vectors.bin"
_IDS_NAME = "ids.bin"
_METADATA_NAME = "metadata.bin"
_DELETED_NAME = "deleted.bin"


class PersistenceManager:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._collections_dir = self._root / _COLLECTIONS_DIR

        self._initialize_database()

    def save_collection(
        self,
        config: CollectionConfig,
        snapshot: StorageSnapshot,
    ) -> None:
        collection_dir = self._collection_dir(config.name)
        snapshots_dir = collection_dir / _SNAPSHOTS_DIR

        collection_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        generation = self._next_generation(snapshots_dir)

        final_snapshot_dir = snapshots_dir / generation
        temporary_snapshot_dir = snapshots_dir / (
            f".tmp-{generation}-{uuid.uuid4().hex}"
        )

        temporary_snapshot_dir.mkdir()

        try:
            self._write_snapshot(
                temporary_snapshot_dir,
                config,
                snapshot,
            )

            self._fsync_directory(temporary_snapshot_dir)

            os.replace(
                temporary_snapshot_dir,
                final_snapshot_dir,
            )

            self._fsync_directory(snapshots_dir)

            self._publish_current(
                collection_dir,
                generation,
            )

        except Exception:
            if temporary_snapshot_dir.exists():
                shutil.rmtree(
                    temporary_snapshot_dir,
                    ignore_errors=True,
                )
            raise

    def load_collection(
        self,
        name: str,
    ) -> tuple[CollectionConfig, StorageSnapshot]:
        collection_dir = self._collection_dir(name)

        if not collection_dir.is_dir():
            raise KeyError(f"Collection {name!r} does not exist")

        current_path = collection_dir / _CURRENT_NAME

        if not current_path.is_file():
            raise CorruptDataError(
                f"Collection {name!r} has no CURRENT snapshot"
            )

        try:
            generation = current_path.read_text(
                encoding="ascii"
            ).strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise CorruptDataError(
                f"Collection {name!r} has unreadable CURRENT"
            ) from exc

        if not self._is_generation_name(generation):
            raise CorruptDataError(
                f"Collection {name!r} has invalid CURRENT"
            )

        snapshot_dir = (
            collection_dir
            / _SNAPSHOTS_DIR
            / generation
        )

        if not snapshot_dir.is_dir():
            raise CorruptDataError(
                f"Snapshot {generation!r} does not exist "
                f"for collection {name!r}"
            )

        config, snapshot = self._read_snapshot(snapshot_dir)

        if config.name != name:
            raise CorruptDataError(
                f"Snapshot collection name {config.name!r} does not match {name!r}"
            )

        return config, snapshot

    def list_collections(self) -> list[str]:
        result: list[str] = []

        for path in self._collections_dir.iterdir():
            if not path.is_dir():
                continue

            if (path / _CURRENT_NAME).is_file():
                result.append(path.name)

        return sorted(result)

    def drop_collection(self, name: str) -> None:
        collection_dir = self._collection_dir(name)

        if not collection_dir.exists():
            raise KeyError(f"Collection {name!r} does not exist")

        trash_name = (
            f".deleted-{name}-{uuid.uuid4().hex}"
        )
        trash_path = self._collections_dir / trash_name

        os.replace(collection_dir, trash_path)
        self._fsync_directory(self._collections_dir)

        shutil.rmtree(trash_path, ignore_errors=True)

    def _initialize_database(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._collections_dir.mkdir(exist_ok=True)

        manifest_path = self._root / _MANIFEST_NAME

        if manifest_path.exists():
            self._validate_manifest(manifest_path)
            return

        temporary_path = (
            self._root
            / f".manifest-{uuid.uuid4().hex}.tmp"
        )

        manifest = {
            "format": "vectorch",
            "format_version": FORMAT_VERSION,
        }

        try:
            self._write_json(
                temporary_path,
                manifest,
            )

            os.replace(
                temporary_path,
                manifest_path,
            )

            self._fsync_directory(self._root)

        finally:
            temporary_path.unlink(missing_ok=True)

    def _validate_manifest(
        self,
        path: Path,
    ) -> None:
        data = self._read_json(path)

        if data.get("format") != "vectorch":
            raise CorruptDataError(
                "Invalid Vectorch database manifest"
            )

        if data.get("format_version") != FORMAT_VERSION:
            raise CorruptDataError(
                "Unsupported Vectorch database format version"
            )

    def _write_snapshot(
        self,
        directory: Path,
        config: CollectionConfig,
        snapshot: StorageSnapshot,
    ) -> None:
        count = len(snapshot.ids)

        if snapshot.vectors.shape != (
            count,
            config.dimension,
        ):
            raise ValueError(
                "Snapshot vector shape does not match config"
            )

        if len(snapshot.metadata) != count:
            raise ValueError(
                "Snapshot metadata count does not match IDs"
            )

        if snapshot.deleted.shape != (count,):
            raise ValueError(
                "Snapshot deleted mask does not match IDs"
            )

        config_data: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "name": config.name,
            "dimension": config.dimension,
            "metric": config.metric.value,
            "index_type": config.index_type.value,
            "index_params": config.index_params,
            "vector_count": count,
            "vector_dtype": "float32",
            "byte_order": "little",
        }

        self._write_json(
            directory / _CONFIG_NAME,
            config_data,
        )

        write_vectors(
            directory / _VECTORS_NAME,
            snapshot.vectors,
        )
        write_ids(
            directory / _IDS_NAME,
            snapshot.ids,
        )
        write_metadata(
            directory / _METADATA_NAME,
            snapshot.metadata,
        )
        write_deleted(
            directory / _DELETED_NAME,
            snapshot.deleted,
        )

        self._fsync_file(directory / _VECTORS_NAME)
        self._fsync_file(directory / _IDS_NAME)
        self._fsync_file(directory / _METADATA_NAME)
        self._fsync_file(directory / _DELETED_NAME)

    def _read_snapshot(
        self,
        directory: Path,
    ) -> tuple[CollectionConfig, StorageSnapshot]:
        config_path = directory / _CONFIG_NAME

        if not config_path.is_file():
            raise CorruptDataError(
                "Snapshot has no config.json"
            )

        data = self._read_json(config_path)

        self._validate_collection_config(data)

        try:
            name = data["name"]
            dimension = data["dimension"]
            count = data["vector_count"]

            config = CollectionConfig(
                name=name,
                dimension=dimension,
                metric=Metric(data["metric"]),
                index_type=IndexType(data["index_type"]),
                index_params=data["index_params"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptDataError(
                "Invalid collection config"
            ) from exc

        required_files = (
            _VECTORS_NAME,
            _IDS_NAME,
            _METADATA_NAME,
            _DELETED_NAME,
        )

        for filename in required_files:
            if not (directory / filename).is_file():
                raise CorruptDataError(
                    f"Snapshot is missing {filename}"
                )

        try:
            snapshot = StorageSnapshot(
                vectors=read_vectors(
                    directory / _VECTORS_NAME,
                    count=count,
                    dimension=dimension,
                ),
                ids=read_ids(
                    directory / _IDS_NAME,
                    count=count,
                ),
                metadata=read_metadata(
                    directory / _METADATA_NAME,
                    count=count,
                ),
                deleted=read_deleted(
                    directory / _DELETED_NAME,
                    count=count,
                ),
            )
        except OSError as exc:
            raise CorruptDataError(
                "Failed to read collection snapshot"
            ) from exc

        return config, snapshot

    def _validate_collection_config(
        self,
        data: dict[str, Any],
    ) -> None:
        if data.get("format_version") != FORMAT_VERSION:
            raise CorruptDataError(
                "Unsupported collection format version"
            )

        if data.get("vector_dtype") != "float32":
            raise CorruptDataError(
                "Unsupported vector dtype"
            )

        if data.get("byte_order") != "little":
            raise CorruptDataError(
                "Unsupported vector byte order"
            )

        name = data.get("name")
        dimension = data.get("dimension")
        count = data.get("vector_count")
        index_params = data.get("index_params")

        try:
            validate_collection_name(name)
        except (TypeError, ValueError) as exc:
            raise CorruptDataError("Invalid collection name") from exc

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise CorruptDataError(
                "Invalid collection dimension"
            )

        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise CorruptDataError(
                "Invalid vector count"
            )

        if not isinstance(index_params, dict):
            raise CorruptDataError(
                "Invalid index parameters"
            )

    def _publish_current(
        self,
        collection_dir: Path,
        generation: str,
    ) -> None:
        temporary_path = (
            collection_dir
            / f".CURRENT-{uuid.uuid4().hex}.tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="ascii",
            ) as file:
                file.write(generation)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            os.replace(
                temporary_path,
                collection_dir / _CURRENT_NAME,
            )

            self._fsync_directory(collection_dir)

        finally:
            temporary_path.unlink(missing_ok=True)

    def _next_generation(
        self,
        snapshots_dir: Path,
    ) -> str:
        maximum = 0

        for path in snapshots_dir.iterdir():
            if (
                path.is_dir()
                and self._is_generation_name(path.name)
            ):
                maximum = max(
                    maximum,
                    int(path.name),
                )

        return f"{maximum + 1:016d}"

    @staticmethod
    def _is_generation_name(value: str) -> bool:
        return (
            len(value) == 16
            and value.isascii()
            and value.isdigit()
        )

    def _collection_dir(
        self,
        name: str,
    ) -> Path:
        validate_collection_name(name)
        return self._collections_dir / name

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict[str, Any]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(
                    file,
                    parse_constant=PersistenceManager._reject_json_constant,
                )
        except (
            OSError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            raise CorruptDataError(
                f"Invalid JSON file: {path.name}"
            ) from exc

        if not isinstance(data, dict):
            raise CorruptDataError(
                f"{path.name} must contain a JSON object"
            )

        return data

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as file:
            os.fsync(file.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)

        try:
            os.fsync(fd)
        finally:
            os.close(fd)
