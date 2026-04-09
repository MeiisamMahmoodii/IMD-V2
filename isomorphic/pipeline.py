from isomorphic.extractors.mean_pooling import MeanPoolingExtractor
from isomorphic.alignment_utils import ProcrustesAligner
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
