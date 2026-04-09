import importlib

EXPERIMENTS = [
"experiments.exp_001_anchor_strategy_comparison",
"experiments.exp_002_extraction_methods",
"experiments.exp_003_cross_dataset_alignment",
"experiments.exp_004_model_pairs",
"experiments.exp_005_alignment_stability",
"experiments.exp_006_scalability",
"experiments.exp_007_semantic_preservation",
"experiments.exp_008_statistical_significance",
]

def main():
    for exp in EXPERIMENTS:
        importlib.import_module(exp).run()

if __name__ == "__main__":
    main()
