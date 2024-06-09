from pathlib import Path

from gorillatracker.data.contrastive_sampler import ContrastiveClassSampler
from gorillatracker.data.nlet import BasicKFoldDataset, BasicDataset


class CXLDataset(BasicDataset):
    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveClassSampler:
        assert "cxl" in base_dir.name, f"Expected 'cxl' in {base_dir.name}, are you sure this is the right dataset?"
        return super().create_contrastive_sampler(base_dir)


class KFoldCXLDataset(BasicKFoldDataset, CXLDataset):
    pass
