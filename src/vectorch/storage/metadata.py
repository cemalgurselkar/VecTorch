from typing import Any


class MetadataStorage:
    def __init__(self) -> None:
        self._data: dict[int, dict[str, Any]] = {}
        
    def set(self, internal_id: int, metadata: dict[str, Any]) -> None:
        if internal_id < 0:
            raise ValueError("Internal ID must be a non-negative integer.")
        
        self._data[internal_id] = metadata
    
    def get(self, internal_id: int) -> dict[str, Any]:
        return self._data.get(internal_id, {})
    
    def get_many(self, internal_ids: list[int]) -> list[dict[str, Any] | None]:
        return [self._data.get(internal_id, None) for internal_id in internal_ids]
    
    def delete(self, internal_id: int) -> None:
        self._data.pop(internal_id, None)
    
    def __len__(self) -> int:
        return len(self._data)