from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class VectorExtractor(ABC):
    def __init__(self, model_name: str, embedding_dim: int = 3584):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    @abstractmethod
    def extract(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def extract_batch(self, texts: list[str]) -> np.ndarray:
        arr = [self.extract(t) for t in texts]
        return np.vstack(arr) if arr else np.zeros((0, self.embedding_dim), dtype=float)
