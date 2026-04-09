from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from experiments.analysis.statistical_summary import main
except ModuleNotFoundError:
    def main():
        script = ROOT / "experiments" / "analysis" / "statistical_summary.py"
        namespace = runpy.run_path(str(script), run_name="__main__")
        if "main" in namespace and callable(namespace["main"]):
            namespace["main"]()

if __name__ == "__main__":
    main()
