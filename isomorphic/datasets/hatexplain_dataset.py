from .base_dataset import BaseDataset

class HateXplainDataset(BaseDataset):
    def load(self) -> list[str]:
        limit = int(self.config.get("limit", 100))
        self.seeds = [f"hatexplain seed sample {i} about social bias and toxicity" for i in range(limit)]
        self.metadata = {"source": "hatexplain", "license": "refer dataset license"}
        return self.seeds
