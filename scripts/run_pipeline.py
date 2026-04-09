from pathlib import Path
import sys
import json
import argparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isomorphic.config import ConfigManager
from isomorphic.pipeline import IsomorphicPipeline
from isomorphic.preprocessor import PreprocessingPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["real", "scaffold"],
        default="real",
        help="real = real HF dataset/model execution, scaffold = lightweight placeholder flow",
    )
    args = parser.parse_args()

    cfg = ConfigManager.load_yaml("config/default.yaml")
    datasets_cfg = ConfigManager.load_yaml("config/datasets.yaml").get("datasets", [])
    output_root = cfg.get("output_dir", "experiments/results")
    if args.mode == "real":
        from isomorphic.real_pipeline import RealPipeline

        models_cfg = ConfigManager.load_yaml("config/models.yaml")
        stats = RealPipeline(models_cfg=models_cfg, output_root=output_root).run(
            datasets_cfg=datasets_cfg,
            threshold=float(cfg.get("wasserstein_threshold", 0.5)),
            min_words=int(cfg.get("generation_min_words", 8)),
            max_words=int(cfg.get("generation_max_words", 20)),
            max_attempts=int(cfg.get("generation_max_attempts", 5)),
            output_path=cfg.get("final_dataset_output", "data/processed/final_dataset.json"),
        )
    else:
        PreprocessingPipeline(output_root=output_root).run(datasets_cfg)
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
                candidate_texts.append(source)

        # Keep scaffold mode lightweight and deterministic (no model downloads).
        pipeline = IsomorphicPipeline("dummy", output_root=output_root)
        stats = pipeline.build_final_dataset(
            source_texts=source_texts,
            candidate_texts=candidate_texts,
            threshold=float(cfg.get("wasserstein_threshold", 0.5)),
            output_path=cfg.get("final_dataset_output", "data/processed/final_dataset.json"),
        )
    print(json.dumps(stats, indent=2))
