import numpy as np
from .base_extractor import VectorExtractor
from .mean_pooling import MeanPoolingExtractor
from .last_token import LastTokenExtractor

class HybridExtractor(VectorExtractor):
    def __init__(self, model_name: str, embedding_dim: int = 3584):
        super().__init__(model_name, embedding_dim)
        self.mean = MeanPoolingExtractor(model_name, embedding_dim)
        self.last = LastTokenExtractor(model_name, embedding_dim)

    def extract(self, text: str) -> np.ndarray:
        vec = 0.5 * self.mean.extract(text) + 0.5 * self.last.extract(text)
        return vec / (np.linalg.norm(vec) + 1e-12)
