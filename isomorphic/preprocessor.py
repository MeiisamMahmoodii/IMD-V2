from pathlib import Path
import json
from isomorphic.loader import DatasetLoader
from isomorphic.text_normalization import normalize_text
from isomorphic.utils.logger import ResultsLogger

class PreprocessingPipeline:
    def __init__(self, output_root: str = "experiments/results"):
        self.results = ResultsLogger(output_root)

    def run(self, dataset_configs: list[dict]) -> dict:
        rows = []
        summary = {}
        for cfg in dataset_configs:
            ds = DatasetLoader.load(cfg["name"], cfg)
            out = Path("data/processed") / f"{cfg['name']}_processed.json"
            ds.save_processed(out)
            processed = json.loads(out.read_text(encoding="utf-8"))
            for row in processed:
                row["normalized_seed"] = normalize_text(row["seed"])
            out.write_text(json.dumps(processed, indent=2), encoding="utf-8")
            stat = ds.statistics()
            rows.append(
                {
                    "dataset": cfg["name"],
                    "total_seeds": stat["total_seeds"],
                    "avg_length": stat["avg_length"],
                    "normalized": True,
                    "processed_path": str(out),
                }
            )
            summary[cfg["name"]] = stat
        self.results.write_csv("preprocessing_report.csv", rows)
        self.results.write_markdown("preprocessing_summary.md", "# Preprocessing Summary\n")
        return summary
