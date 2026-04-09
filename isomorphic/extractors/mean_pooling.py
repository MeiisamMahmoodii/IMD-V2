import numpy as np
from .base_extractor import VectorExtractor

class MeanPoolingExtractor(VectorExtractor):
    def extract(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.embedding_dim, dtype=float)
        rng = np.random.default_rng(sum(ord(c) for c in text))
        vec = rng.normal(0.0, 1.0, self.embedding_dim)
        return vec / (np.linalg.norm(vec) + 1e-12)
