import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from isomorphic.extractors.mean_pooling import MeanPoolingExtractor
from isomorphic.alignment_utils import ProcrustesAligner
from isomorphic.text_normalization import normalize_text
from isomorphic.utils.logger import ResultsLogger

class IsomorphicPipeline:
    def __init__(self, embedding_model: str, output_root: str = "experiments/results"):
        self.extractor = MeanPoolingExtractor(embedding_model)
        self.results = ResultsLogger(output_root)

    def run_alignment(self, source_texts: list[str], target_texts: list[str], experiment_id: str) -> dict:
        x = self.extractor.extract_batch(source_texts)
        y = self.extractor.extract_batch(target_texts)
        q, error = ProcrustesAligner.align(x, y)
        verify = ProcrustesAligner.verify_orthogonality(q)
        payload = {"experiment_id": experiment_id, "alignment": {"frobenius_error": error, **verify}, "num_samples": len(source_texts)}
        self.results.log_experiment(experiment_id, payload)
        self.results.write_csv("alignment_report.csv", [{"experiment_id": experiment_id, "frobenius_error": error}])
        return payload

    def build_final_dataset(
        self,
        source_texts: list[str],
        candidate_texts: list[str],
        threshold: float = 0.5,
        output_path: str = "data/processed/final_dataset.json",
    ) -> dict:
        if len(source_texts) != len(candidate_texts):
            raise ValueError("source_texts and candidate_texts must have same length")

        rows = []
        kept = []
        distances = []
        for idx, (src, cand) in enumerate(zip(source_texts, candidate_texts)):
            src_n = normalize_text(src)
            cand_n = normalize_text(cand)
            src_vec = self.extractor.extract(src_n)
            cand_vec = self.extractor.extract(cand_n)
            dist = float(wasserstein_distance(src_vec, cand_vec))
            keep = dist <= threshold
            row = {
                "row_id": idx,
                "source_text": src,
                "candidate_text": cand,
                "normalized_source": src_n,
                "normalized_candidate": cand_n,
                "wasserstein_distance": dist,
                "accepted": keep,
            }
            rows.append(row)
            distances.append(dist)
            if keep:
                kept.append(row)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(kept, indent=2), encoding="utf-8")

        stats = {
            "threshold": threshold,
            "total_rows": len(rows),
            "accepted_rows": len(kept),
            "acceptance_rate": (len(kept) / len(rows)) if rows else 0.0,
            "distance_mean": float(np.mean(distances)) if distances else 0.0,
            "distance_std": float(np.std(distances)) if distances else 0.0,
            "output_path": str(out),
        }
        self.results.write_csv("final_dataset_filtering_stats.csv", [stats])
        self.results.write_csv(
            "final_dataset_distances.csv",
            [
                {
                    "row_id": r["row_id"],
                    "wasserstein_distance": r["wasserstein_distance"],
                    "accepted": r["accepted"],
                }
                for r in rows
            ],
        )
        self.results.write_markdown(
            "final_dataset_summary.md",
            (
                "# Final Dataset Summary\n\n"
                f"- Threshold: {threshold}\n"
                f"- Total rows: {stats['total_rows']}\n"
                f"- Accepted rows: {stats['accepted_rows']}\n"
                f"- Acceptance rate: {stats['acceptance_rate']:.4f}\n"
                f"- Mean distance: {stats['distance_mean']:.6f}\n"
            ),
        )
        return stats
