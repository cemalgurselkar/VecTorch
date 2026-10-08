import json
import math
from collections.abc import Sequence
from typing import Any, Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray


class _TorchDevice(Protocol):
    type: str
    
class _TorchTensor(Protocol):
    @property
    def device(self) -> _TorchDevice:
        ...
    
    @property
    def requires_grad(self) -> bool:
        ...
        
    def detach(self) -> "_TorchTensor":
        ...
    
    def numpy(self) -> NDArray[np.float32]:
        ...

vector_input = NDArray[Any] | Sequence[float] | Any

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def validate_collection_name(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Collection name must be a non-empty string")

    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Invalid collection name: {name!r}")

    return name


def validate_external_id(external_id: object) -> str | int:
    if isinstance(external_id, bool) or not isinstance(external_id, (str, int)):
        raise TypeError("External ID must be a string or integer")

    if isinstance(external_id, int) and not (
        _INT64_MIN <= external_id <= _INT64_MAX
    ):
        raise ValueError("Integer external ID must fit signed int64")

    return external_id


def validate_metadata(metadata: object) -> dict[str, Any] | None:
    if metadata is None:
        return None

    if not isinstance(metadata, dict):
        raise TypeError("Metadata must be a dictionary or None")

    try:
        _validate_json_value(metadata, active_containers=set())
        json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeEncodeError,
        ValueError,
    ) as exc:
        raise ValueError("Metadata must be JSON-compatible") from exc

    return metadata


def _validate_json_value(value: object, active_containers: set[int]) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Metadata numbers must be finite")
        return

    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in active_containers:
            raise ValueError("Metadata cannot contain cycles")

        active_containers.add(identity)
        try:
            if isinstance(value, list):
                for item in value:
                    _validate_json_value(item, active_containers)
            else:
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("Metadata object keys must be strings")
                    _validate_json_value(item, active_containers)
        finally:
            active_containers.remove(identity)
        return

    raise ValueError(
        f"Unsupported metadata value type: {type(value).__name__}"
    )

def validate_vector(vector: vector_input, *, dimension: int, name: str = "vector") -> NDArray[np.float32]:
    """
    Internal invariant:
     -np.ndarray
     -dtype=np.float32
     -shape=(dimension,)
     C-contiguous
     -finite values only
    """
    
    if _is_torch_tensor(vector):
        if vector.device.type != "cpu":
            raise ValueError(f"vector must be on CPU, but is on {vector.device.type}")

        if vector.requires_grad:
            vector = vector.detach()
        
        vector = vector.numpy()
    try:
        array = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid {name} of type {type(vector)}") from e
    
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, but is {array.ndim}-dimensional")
    
    if array.shape[0] != dimension:
        raise ValueError(f"{name} must have length {dimension}, but has length {array.shape[0]}")
    
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    
    return array
    
        
def _is_torch_tensor(value: object) -> TypeGuard[_TorchTensor]:
    cls = type(value)

    return (cls.__module__.startswith("torch") and cls.__name__ == "Tensor")
