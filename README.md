# IMD-V2 (IsomorphicDataSet V2)

Research-oriented framework for testing latent-space isomorphism across LLMs using multi-dataset pipelines, vector extraction, Procrustes alignment, and reproducible experiment reporting.

## What This Repo Contains

- `isomorphic/` core package (config, datasets, extractors, alignment, pipeline)
- `experiments/` experiment runners (`exp_001` to `exp_008`) and generated outputs
- `config/` model, dataset, and experiment configuration files
- `scripts/` setup, run, stats, and reproducibility helpers
- `tests/` unit, integration, performance, and statistics tests
- `planning/` source planning documents used to define implementation scope

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

## Run Pipeline and Experiments

```bash
python scripts/setup_datasets.py
python scripts/run_experiments.py
python scripts/compute_statistics.py
python scripts/generate_figures.py
```

## Reproducibility and Archival

- Fixed-seed workflow is documented in `docs/REPRODUCIBILITY.md`
- Full regeneration script: `scripts/regenerate_all_results.py`
- Results archival script: `scripts/archive_results.py`

## Repository URL

- GitHub: https://github.com/MeiisamMahmoodii/IMD-V2
