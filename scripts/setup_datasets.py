from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isomorphic.config import ConfigManager
from isomorphic.preprocessor import PreprocessingPipeline

def main():
    cfg = ConfigManager.load_yaml("config/datasets.yaml")
    PreprocessingPipeline().run(cfg.get("datasets", []))

if __name__ == "__main__":
    main()
