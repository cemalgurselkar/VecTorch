import numpy as np
from numpy.typing import NDArray

from ..types import Metric
from .base import DistanceKernel


class NumpyKernel(DistanceKernel):
    def __init__(self, metric: Metric) -> None:
        self._metric = metric
        
    def compute(self, query: NDArray[np.float32], vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        
        if self._metric == Metric.L2:
            diff = vectors - query # temporary allocation
            return np.sum(diff * diff, axis=1)
        
        if self._metric == Metric.DOT:
            return vectors @ query
        
        if self._metric == Metric.COSINE:
            dot = vectors @ query
            
            vector_norms = np.linalg.norm(vectors, axis=1)
            query_norms = np.linalg.norm(query)
            
            denominator = vector_norms * query_norms
            
            return dot / np.maximum(denominator, 1e-6)
        
        raise ValueError(f"Invalid metric: {self._metric}")