from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isomorphic.config import ConfigManager
from isomorphic.pipeline import IsomorphicPipeline
from isomorphic.preprocessor import PreprocessingPipeline

if __name__ == "__main__":
    cfg = ConfigManager.load_yaml("config/default.yaml")
    datasets_cfg = ConfigManager.load_yaml("config/datasets.yaml").get("datasets", [])
    PreprocessingPipeline(output_root=cfg.get("output_dir", "experiments/results")).run(datasets_cfg)

    source_texts = []
    candidate_texts = []
    for dataset in datasets_cfg:
        path = Path("data/processed") / f"{dataset['name']}_processed.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            source = row.get("seed", "")
            source_texts.append(source)
            # Placeholder for rewritten candidate text; in production this is model-generated text.
            candidate_texts.append(source)

    pipeline = IsomorphicPipeline(cfg.get("embedding_model", "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2"), output_root=cfg.get("output_dir", "experiments/results"))
    stats = pipeline.build_final_dataset(
        source_texts=source_texts,
        candidate_texts=candidate_texts,
        threshold=float(cfg.get("wasserstein_threshold", 0.5)),
        output_path=cfg.get("final_dataset_output", "data/processed/final_dataset.json"),
    )
    print(json.dumps(stats, indent=2))
