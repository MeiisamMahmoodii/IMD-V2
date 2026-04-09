import importlib
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS = [
    "exp_001_anchor_strategy_comparison",
    "exp_002_extraction_methods",
    "exp_003_cross_dataset_alignment",
    "exp_004_model_pairs",
    "exp_005_alignment_stability",
    "exp_006_scalability",
    "exp_007_semantic_preservation",
    "exp_008_statistical_significance",
]

def main():
    for exp in EXPERIMENTS:
        module_name = f"experiments.{exp}"
        try:
            importlib.import_module(module_name).run()
            continue
        except ModuleNotFoundError:
            pass

        script_path = ROOT / "experiments" / f"{exp}.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Experiment script not found: {script_path}")
        namespace = runpy.run_path(str(script_path), run_name="__main__")
        if "run" in namespace and callable(namespace["run"]):
            namespace["run"]()

if __name__ == "__main__":
    main()
