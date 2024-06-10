from pathlib import Path

from gorillatracker.data.contrastive_sampler import ContrastiveClassSampler
from gorillatracker.data.nlet import BasicDataset


class BristolDataset(BasicDataset):
    def create_contrastive_sampler(self, base_dir: Path) -> ContrastiveClassSampler:
        assert (
            "bristol" in base_dir.name
        ), f"Expected 'bristol' in {base_dir.name}, are you sure this is the right dataset?"
        return super().create_contrastive_sampler(base_dir)
