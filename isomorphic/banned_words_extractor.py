from dataclasses import dataclass
import re

STOPWORDS = {"the","and","this","that","with","from","have","will","into","about","there"}
PROMPT_TEMPLATE = "Extract 5-7 semantic words from seed text.\nSEED TEXT:\n{seed}"

@dataclass
class BannedWordsExtractor:
    model_id: str = "Qwen2.5-72B-Instruct-abliterated"
    min_chars: int = 4

    def build_prompt(self, seed: str) -> str:
        return PROMPT_TEMPLATE.format(seed=seed)

    def extract(self, seed: str, top_k: int = 6) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z\-']+", seed.lower())
        keep = [w for w in words if len(w) >= self.min_chars and w not in STOPWORDS]
        uniq = []
        for w in keep:
            if w not in uniq:
                uniq.append(w)
        return uniq[: max(5, min(top_k, 7))]
