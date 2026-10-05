"""
external ID  ↔  internal ID

"doc_abc"    ↔  0
"doc_xyz"    ↔  1

internal_idler doğrudan list index olabilir. o yüzden 2 tane dictionary kullanmadım

"""

class IDMap:
    def __init__(self) -> None:
        self._external_to_internal: dict[str | int, int] = {}
        self._internal_to_external: list[str | int] = []
    
    def add(self, external_id: str | int) -> int:
        if external_id in self._external_to_internal:
            raise ValueError(f"External ID {external_id} already exists.")
        
        internal_id = len(self._external_to_internal)
        self._external_to_internal[external_id] = internal_id
        self._internal_to_external.append(external_id)
        return internal_id
    
    def internal(self, external_id: str | int) -> int:
        try:
            return self._external_to_internal[external_id]
        except KeyError:
            raise KeyError(f"External ID {external_id} does not exist.") from None
    
    def external(self, internal_id: int) -> str | int:
        if internal_id < 0 or internal_id >= len(self._internal_to_external):
            raise IndexError(f"Internal ID {internal_id} is out of bounds.")
        
        return self._internal_to_external[internal_id]
    
    def contains(self, external_id: str | int) -> bool:
        return external_id in self._external_to_internal
    
    def __len__(self) -> int:
        return len(self._external_to_internal)
    