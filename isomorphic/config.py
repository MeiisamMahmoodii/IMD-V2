from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class ModelConfig:
    name: str
    model_id: str
    purpose: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 8
    max_length: int = 2048
    load_in_8bit: bool = False

@dataclass
class DatasetConfig:
    name: str
    source: str
    split: str = "train"
    limit: int | None = None

@dataclass
class ExtractionConfig:
    method: str = "mean_pooling"
    embedding_dim: int = 3584

@dataclass
class AlignmentConfig:
    method: str = "procrustes_svd"
    center_data: bool = True

@dataclass
class ExperimentConfig:
    name: str
    models: list[ModelConfig]
    datasets: list[DatasetConfig]
    extraction: ExtractionConfig
    alignment: AlignmentConfig
    random_seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("experiments/results"))

class ConfigManager:
    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def save_yaml(config: dict[str, Any], path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
