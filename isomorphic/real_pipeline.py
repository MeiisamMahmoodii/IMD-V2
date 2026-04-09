from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from scipy.stats import wasserstein_distance

from isomorphic.alignment_utils import ProcrustesAligner
from isomorphic.banned_words_extractor import BannedWordsExtractor
from isomorphic.extractors.mean_pooling import MeanPoolingExtractor
from isomorphic.text_normalization import normalize_text
from isomorphic.utils.logger import ResultsLogger


DATASET_ID_MAP = {
    "toxigen": "skg/toxigen-data",
    "hatexplain": "hatexplain",
    "sbic": "social_bias_frames",
    "ethos": "ethos",
}


def _pick_text(row: dict[str, Any]) -> str:
    for key in ("text", "comment_text", "post", "sentence"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key]
    if "post_tokens" in row and isinstance(row["post_tokens"], list):
        return " ".join(str(x) for x in row["post_tokens"])
    if "tokens" in row and isinstance(row["tokens"], list):
        return " ".join(str(x) for x in row["tokens"])
    return ""


def _word_count_ok(text: str, min_words: int, max_words: int) -> bool:
    wc = len([w for w in text.strip().split(" ") if w])
    return min_words <= wc <= max_words


def _contains_forbidden(text: str, forbidden: list[str]) -> bool:
    lowered = text.lower()
    return any(f.lower() in lowered for f in forbidden)


@dataclass
class HFRewriter:
    model_id: str

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Real generation requires torch+transformers. Install with pip install -e .[ml]"
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        seed: str,
        forbidden_words: list[str],
        min_words: int,
        max_words: int,
        max_attempts: int,
    ) -> tuple[str | None, int]:
        prompt = (
            f"WORD COUNT REQUIREMENT: {min_words}-{max_words} words only.\n"
            f"Rewrite into one sentence, keep meaning, avoid forbidden words.\n"
            f"Forbidden words: {', '.join(forbidden_words)}\n"
            f"Input: {seed}\n"
            "Output only rewritten sentence."
        )
        for attempt in range(1, max_attempts + 1):
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            out = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            text = self._tokenizer.decode(out[0], skip_special_tokens=True)
            candidate = text.split("Output only rewritten sentence.")[-1].strip().split("\n")[0].strip()
            if candidate and _word_count_ok(candidate, min_words, max_words) and not _contains_forbidden(
                candidate, forbidden_words
            ):
                return candidate, attempt
        return None, max_attempts


class RealPipeline:
    def __init__(
        self,
        models_cfg: dict[str, Any],
        output_root: str = "experiments/results",
        embedding_model_id: str | None = None,
    ):
        self.models_cfg = models_cfg
        self.results = ResultsLogger(output_root)
        self.embedding_model_id = embedding_model_id or models_cfg["embedding"]["primary"]["model_id"]
        self.embedding_extractor = MeanPoolingExtractor(self.embedding_model_id)
        self.banned_extractor = BannedWordsExtractor(models_cfg["banned_words_extractor"]["model_id"])

    def _load_dataset_rows(self, name: str, split: str, limit: int) -> list[str]:
        if name == "jigsaw":
            raise RuntimeError("Jigsaw requires Kaggle source; add local loader path for production run.")
        dataset_id = DATASET_ID_MAP.get(name)
        if not dataset_id:
            raise ValueError(f"No dataset_id configured for dataset: {name}")
        ds = load_dataset(dataset_id, split=split)
        if limit and len(ds) > limit:
            ds = ds.select(range(limit))
        texts = []
        for row in ds:
            text = _pick_text(row)
            if text:
                texts.append(text)
        return texts

    def run(
        self,
        datasets_cfg: list[dict[str, Any]],
        threshold: float = 0.5,
        min_words: int = 8,
        max_words: int = 20,
        max_attempts: int = 5,
        output_path: str = "data/processed/final_dataset.json",
        rewriting_model_names: list[str] | None = None,
    ) -> dict[str, Any]:
        all_source_rows: list[dict[str, Any]] = []
        for cfg in datasets_cfg:
            rows = self._load_dataset_rows(cfg["name"], cfg.get("split", "train"), int(cfg.get("limit", 100)))
            for text in rows:
                norm = normalize_text(text)
                banned = self.banned_extractor.extract(norm)
                all_source_rows.append(
                    {
                        "dataset": cfg["name"],
                        "source_text": text,
                        "normalized_source": norm,
                        "forbidden_words": banned,
                    }
                )

        final_rows: list[dict[str, Any]] = []
        align_rows: list[dict[str, Any]] = []
        rewriting_pool = self.models_cfg.get("rewriting", {})
        if rewriting_model_names:
            rewriting_pool = {
                name: rewriting_pool[name]
                for name in rewriting_model_names
                if name in rewriting_pool
            }
        for rewriter_name, spec in rewriting_pool.items():
            model_id = spec["model_id"]
            rewriter = HFRewriter(model_id)
            source_vecs = []
            rewrite_vecs = []
            accepted = 0
            attempted = 0

            for row in all_source_rows:
                attempted += 1
                generated, attempts_used = rewriter.generate(
                    seed=row["normalized_source"],
                    forbidden_words=row["forbidden_words"],
                    min_words=min_words,
                    max_words=max_words,
                    max_attempts=max_attempts,
                )
                if not generated:
                    continue
                src_vec = self.embedding_extractor.extract(row["normalized_source"])
                gen_norm = normalize_text(generated)
                gen_vec = self.embedding_extractor.extract(gen_norm)
                dist = float(wasserstein_distance(src_vec, gen_vec))
                keep = dist <= threshold
                if keep:
                    accepted += 1
                    final_rows.append(
                        {
                            "dataset": row["dataset"],
                            "rewriter_model": rewriter_name,
                            "rewriter_model_id": model_id,
                            "source_text": row["source_text"],
                            "rewritten_text": generated,
                            "normalized_source": row["normalized_source"],
                            "normalized_rewritten": gen_norm,
                            "forbidden_words": row["forbidden_words"],
                            "generation_attempts": attempts_used,
                            "wasserstein_distance": dist,
                        }
                    )
                source_vecs.append(src_vec)
                rewrite_vecs.append(gen_vec)

            if source_vecs and rewrite_vecs:
                x = np.vstack(source_vecs)
                y = np.vstack(rewrite_vecs)
                q, error = ProcrustesAligner.align(x, y)
                verify = ProcrustesAligner.verify_orthogonality(q)
                align_rows.append(
                    {
                        "rewriter_model": rewriter_name,
                        "rewriter_model_id": model_id,
                        "rows_attempted": attempted,
                        "rows_accepted": accepted,
                        "acceptance_rate": (accepted / attempted) if attempted else 0.0,
                        "frobenius_error": error,
                        "orthogonality_error": verify["orthogonality_error"],
                        "determinant": verify["determinant"],
                    }
                )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(final_rows, indent=2), encoding="utf-8")

        stats = {
            "total_source_rows": len(all_source_rows),
            "final_rows": len(final_rows),
            "threshold": threshold,
            "output_path": str(out),
        }
        self.results.write_csv("final_dataset_filtering_stats.csv", [stats])
        if align_rows:
            self.results.write_csv("alignment_real_report.csv", align_rows)
        self.results.write_markdown(
            "final_dataset_summary.md",
            (
                "# Final Dataset Summary\n\n"
                f"- Total source rows: {stats['total_source_rows']}\n"
                f"- Final accepted rows: {stats['final_rows']}\n"
                f"- Wasserstein threshold: {threshold}\n"
            ),
        )
        return stats

