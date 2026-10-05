from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class DistanceKernel(ABC):
    @abstractmethod
    def compute(self, query, vectors) -> NDArray[np.float32]:
        ...