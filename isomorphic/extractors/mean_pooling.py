from __future__ import annotations

import numpy as np

from .base_extractor import VectorExtractor

class MeanPoolingExtractor(VectorExtractor):
    def extract(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.embedding_dim, dtype=float)

        # Fast deterministic path used by tests/scaffold.
        if self.model_name == "dummy":
            rng = np.random.default_rng(sum(ord(c) for c in text))
            vec = rng.normal(0.0, 1.0, self.embedding_dim)
            return vec / (np.linalg.norm(vec) + 1e-12)

        tokenizer, model, torch = self._load_hf()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state[0]  # [seq, dim]
            attn = inputs["attention_mask"][0].unsqueeze(-1).float()
            pooled = (hidden * attn).sum(dim=0) / attn.sum(dim=0).clamp(min=1e-9)
        vec = pooled.detach().cpu().numpy().astype(float)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def _load_hf(self):
        if hasattr(self, "_hf_cached"):
            return self._hf_cached
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Real extraction requires torch+transformers. Install with pip install -e .[ml]"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.eval()
        self.embedding_dim = int(model.config.hidden_size)
        self._hf_cached = (tokenizer, model, torch)
        return self._hf_cached
