# NOTE(memben): let's worry about how we parse configs from the yaml file later

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from gorillatracker.ssl_pipeline.data_structures import IndexedCliqueGraph
from gorillatracker.ssl_pipeline.models import TrackingFrameFeature


@dataclass(frozen=True, order=True)
class ContrastiveImage:
    id: int
    image_path: str
    class_label: int

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path)


class ContrastiveSampler(ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> ContrastiveImage:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a different positive sample from the same class."""
        pass

    @abstractmethod
    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        """Return a negative sample from a different class."""
        pass


class CliqueGraphSampler(ContrastiveSampler):
    def __init__(self, graph: IndexedCliqueGraph[ContrastiveImage]):
        self.graph = graph

    def __getitem__(self, idx: int) -> ContrastiveImage:
        return self.graph[idx]

    def __len__(self) -> int:
        return len(self.graph)

    def positive(self, sample: ContrastiveImage) -> ContrastiveImage:
        return self.graph.get_random_clique_member(sample)

    def negative(self, sample: ContrastiveImage) -> ContrastiveImage:
        return self.graph.get_random_adjacent_clique_member(sample)


DB_URI = "postgresql+psycopg2://postgres:DEV_PWD_139u02riowenfgiw4y589wthfn@postgres:5432/postgres"


def DEMO_get_clique_graph() -> IndexedCliqueGraph[ContrastiveImage]:
    engine = create_engine(DB_URI)
    session_cls = sessionmaker(bind=engine)
    with session_cls() as session:
        tracked_features = list(
            session.execute(
                select(TrackingFrameFeature)
                .where(
                    TrackingFrameFeature.cache_path.isnot(None),
                    TrackingFrameFeature.tracking_id.isnot(None),
                    TrackingFrameFeature.feature_type == "body",
                )
                .order_by(TrackingFrameFeature.tracking_id)
            )
            .scalars()
            .all()
        )

        # TODO(memben): tracking_id != label
        contrastive_images = [ContrastiveImage(f.tracking_frame_feature_id, f.cache_path, f.tracking_id) for f in tracked_features]  # type: ignore

        graph = IndexedCliqueGraph(contrastive_images)

        # merge within the same tracking_id
        for i in range(1, len(tracked_features)):
            if tracked_features[i].tracking_id == tracked_features[i - 1].tracking_id:
                graph.merge(contrastive_images[i], contrastive_images[i - 1])

        # partition by tracking_id
        tracking_id_mapper = {f.tracking_id: i for i, f in enumerate(tracked_features)}
        for tracking_id in tracking_id_mapper:
            for other_tracking_id in tracking_id_mapper:
                if tracking_id != other_tracking_id:
                    graph.partition(
                        contrastive_images[tracking_id_mapper[tracking_id]],
                        contrastive_images[tracking_id_mapper[other_tracking_id]],
                    )

        return graph


def WIP_clique_sampler() -> CliqueGraphSampler:
    graph = DEMO_get_clique_graph()
    sampler = CliqueGraphSampler(graph)
    return sampler


if __name__ == "__main__":
    DEMO_get_clique_graph()
