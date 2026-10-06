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