import numpy as np
from .base_extractor import VectorExtractor

class AttentionWeightedExtractor(VectorExtractor):
    def extract(self, text: str) -> np.ndarray:
        toks = text.split()
        if not toks:
            return np.zeros(self.embedding_dim, dtype=float)
        vec = np.zeros(self.embedding_dim, dtype=float)
        denom = 0.0
        for idx, tok in enumerate(toks, start=1):
            rng = np.random.default_rng(sum(ord(c) for c in tok))
            w = idx / len(toks)
            vec += w * rng.normal(0.0, 1.0, self.embedding_dim)
            denom += w
        vec /= denom
        return vec / (np.linalg.norm(vec) + 1e-12)
