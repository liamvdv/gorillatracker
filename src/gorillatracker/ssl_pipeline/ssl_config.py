from dataclasses import dataclass

from gorillatracker.ssl_pipeline.contrastive_sampler import ContrastiveSampler

# dataclass SSLConfig


@dataclass(kw_only=True)  # type: ignore
class SSLConfig:
    tff_selection: str
    n_videos: int
    n_samples: int
    feature_types: list[str]
    min_confidence: float
    split: object

    def get_contrastive_sampler(self) -> ContrastiveSampler:
        return
