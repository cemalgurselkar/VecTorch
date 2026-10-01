from .types import CollectionConfig, Record, SearchResult

class Collection:
    
    def __init__(self, config: CollectionConfig) -> None:
        self._config = config
        
        self._storage = None
        self._index = None
        self._persistence = None
    
    @property
    def name(self) -> str:
        return self._config.name
    
    @property
    def dimension(self) -> int:
        return self._config.dimension
    
    @property
    def metric(self) -> str:
        return self._config.metric.value 
    
    def add(self, ids: list[str | int], vectors, metadata: list[dict] | None = None) -> None:
        raise NotImplementedError
    
    def search(self, query_vectors, top_k: int = 10) -> list[list[SearchResult]]:
        raise NotImplementedError
    
    def delete(self, ids: list[str | int]) -> None:
        raise NotImplementedError
    
    def get(self, ids: list[str | int]) -> list[Record]:
        raise NotImplementedError
    
    def save(self) -> None:
        raise NotImplementedError
    
    def close(self) -> None:
        raise NotImplementedError
    