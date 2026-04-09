import numpy as np
from .base_extractor import VectorExtractor

class LastTokenExtractor(VectorExtractor):
    def extract(self, text: str) -> np.ndarray:
        token = text.split()[-1] if text.split() else ""
        rng = np.random.default_rng(sum(ord(c) for c in token) + len(text))
        vec = rng.normal(0.0, 1.0, self.embedding_dim)
        return vec / (np.linalg.norm(vec) + 1e-12)
