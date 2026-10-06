from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Index(ABC):
    @abstractmethod
    def search(self, query, k) -> tuple[NDArray[np.integer], NDArray[np.float32]]:
        ...