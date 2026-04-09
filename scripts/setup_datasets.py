from isomorphic.config import ConfigManager
from isomorphic.preprocessor import PreprocessingPipeline

def main():
    cfg = ConfigManager.load_yaml("config/datasets.yaml")
    PreprocessingPipeline().run(cfg.get("datasets", []))

if __name__ == "__main__":
    main()
