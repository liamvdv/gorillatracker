from pathlib import Path

from gorillatracker.data.contrastive_sampler import ContrastiveClassSampler
from gorillatracker.data.nlet import BasicDataset, BasicKFoldDataset


class CZooDataset(BasicDataset):
    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveClassSampler:
        assert "czoo" in base_dir.name, f"Expected 'czoo' in {base_dir.name}, are you sure this is the right dataset?"
        return super().create_contrastive_sampler(base_dir)


class CTaiDataset(BasicDataset):
    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveClassSampler:
        assert "ctai" in base_dir.name, f"Expected 'ctai' in {base_dir.name}, are you sure this is the right dataset?"
        return super().create_contrastive_sampler(base_dir)


class KFoldCZooDataset(BasicKFoldDataset, CZooDataset):
    pass


class KFoldCTaiDataset(BasicKFoldDataset, CTaiDataset):
    pass
