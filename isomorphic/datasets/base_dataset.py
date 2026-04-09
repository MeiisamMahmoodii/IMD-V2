from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from isomorphic.banned_words_extractor import BannedWordsExtractor

@dataclass
class StandardSeed:
    seed: str
    forbidden_words: list[str]
    semantic_intent: str
    dataset_source: str
    original_id: str

class BaseDataset(ABC):
    def __init__(self, config: dict[str, Any], extractor: BannedWordsExtractor | None = None):
        self.config = config
        self.name = config.get("name", "unknown")
        self.seeds: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.extractor = extractor or BannedWordsExtractor()

    @abstractmethod
    def load(self) -> list[str]:
        raise NotImplementedError

    def validate(self) -> bool:
        self.seeds = [s for s in self.seeds if isinstance(s, str) and s.strip()]
        return bool(self.seeds)

    def preprocess(self) -> list[dict[str, Any]]:
        if not self.seeds:
            self.load()
        out = []
        for idx, seed in enumerate(self.seeds):
            out.append(StandardSeed(seed=seed, forbidden_words=self.extractor.extract(seed), semantic_intent="toxicity analysis", dataset_source=self.name, original_id=f"{self.name}_{idx}").__dict__)
        return out

    def save_processed(self, output_path: str | Path) -> Path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.preprocess(), indent=2), encoding="utf-8")
        return p

    def statistics(self) -> dict[str, Any]:
        lengths = [len(s.split()) for s in self.seeds]
        return {"dataset": self.name, "total_seeds": len(self.seeds), "avg_length": (sum(lengths)/len(lengths)) if lengths else 0, "min_length": min(lengths) if lengths else 0, "max_length": max(lengths) if lengths else 0, **self.metadata}
