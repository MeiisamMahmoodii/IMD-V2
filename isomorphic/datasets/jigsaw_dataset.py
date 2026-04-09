from .base_dataset import BaseDataset

class JigsawDataset(BaseDataset):
    def load(self) -> list[str]:
        limit = int(self.config.get("limit", 100))
        self.seeds = [f"jigsaw seed sample {i} about social bias and toxicity" for i in range(limit)]
        self.metadata = {"source": "jigsaw", "license": "refer dataset license"}
        return self.seeds
