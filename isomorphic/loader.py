from isomorphic.datasets import ToxiGenDataset, JigsawDataset, HateXplainDataset, SBICDataset, ETHOSDataset

class DatasetLoader:
    REGISTRY = {"toxigen": ToxiGenDataset, "jigsaw": JigsawDataset, "hatexplain": HateXplainDataset, "sbic": SBICDataset, "ethos": ETHOSDataset}

    @classmethod
    def load(cls, dataset_name: str, config: dict):
        if dataset_name not in cls.REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        ds = cls.REGISTRY[dataset_name](config)
        ds.load()
        ds.validate()
        return ds

    @classmethod
    def load_multiple(cls, configs: list[dict]):
        return {cfg["name"]: cls.load(cfg["name"], cfg) for cfg in configs}
