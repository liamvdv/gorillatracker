import pathlib
import torch
import torch.utils.data as data_utils
import tqdm
import faiss
import numpy as np
import networkx as nx
import logging
import pandas as pd
import gorillatracker.model as model
import gorillatracker.datasets.cxl as cxl

logger = logging.getLogger("GT-CurationPipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

embedding_model_settings = {
    "from_scratch": True,
    "loss_mode": "offline/native",
    "weight_decay": 0.5,
    "lr_schedule": "constant",
    "warmup_mode": "constant",
    "warmup_epochs": 0,
    "initial_lr": 1e-5,
    "start_lr": 1e-5,
    "end_lr": 1e-5,
    "max_epochs": 20,
    "beta1": 0.9,
    "beta2": 0.999,
    "embedding_size": 256,
}


class CurationPipeline:
    """
    Pipeline for curation of large datasets. The pipeline follows the following paper with the difference that
    we use L2 distnace instead of cosine similarity for the clustering step.
    https://arxiv.org/pdf/2304.07193.pdf
    """

    def __init__(self, embedding_model_path: pathlib.Path, embedding_model=model.EfficientNetV2Wrapper):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_class = cxl.CXLDataset

        # dataloader settings
        self.batch_size = 32
        self.num_workers = 8

        # curation settings
        self.k_nearest_neighbors = 8  # number of nearest neighbors to consider for clustering
        self.similarity_threshold_self = 0.17  # similarity threshold for self dedublication
        self.number_of_representatives = 1  # number of representatives to keep from each cluster
        self.similarity_threshold_relative = 0.2  # similarity threshold for relative dedublication

        # setup embedding model
        self.embedding_model_path = embedding_model_path
        self.embedding_model = embedding_model(
            model_name_or_path=self.embedding_model_path, **embedding_model_settings
        ).to(self.device)
        self.embedding_model.load_state_dict(torch.load(self.embedding_model_path)["state_dict"])
        self.embedding_model.eval()

        logger.info("CurationPipeline successfully initialized!")

    def curate_patitioned_dataset(self, source: str, destination: str) -> None:
        logger.info("Curating dataset from source: %s to destination: %s", source, destination)
        partitions = ["test", "val"]  # TODO: replace with ["train", "val", "test"]

        embeddings_by_partition = self._get_embeddings_by_partition(source=source, partitions=partitions)
        embeddings_test_df = pd.DataFrame(embeddings_by_partition["test"])
        embeddings_val_df = pd.DataFrame(embeddings_by_partition["val"])
        # embeddings_train_df = pd.DataFrame(embeddings_by_partition["train"])
        embeddings_test_df.to_csv(pathlib.Path(destination) / "test_partition_df.csv.", index=False)
        embeddings_train_df.to_csv(pathlib.Path(destination) / "train_partition_df.csv.", index=False)
        embeddings_val_df.to_csv(pathlib.Path(destination) / "val_partition_df.csv.", index=False)

        print(embeddings_test_df.shape)

    def _get_embeddings_by_partition(
        self, source: str, destination: str, partitions: list[str], use_cache: bool = True, save_embeddings: bool = True
    ) -> dict[str, list]:
        """Retrieves the embeddigns for all partitions given from the source path"""
        embeddings_by_partition = {}
        for partition in partitions:
            logger.info("Gathering embeddings for partion: %s", partition)
            cache_destination = pathlib.Path(destination) / f"{partition}_partition_df.csv"

            if use_cache and cache_destination.exists():
                embeddings_df = pd.read_csv(cache_destination)
            else:
                dataset = self.dataset_class(
                    data_dir=source, partition=partition, transform=self.dataset_class.get_transforms()
                )
                embeddings, paths = self._get_embeddings(dataset)
                embeddings_for_df = []
                for embedding, path in zip(embeddings, paths):
                    embeddings_for_df.append({"embedding": embedding, "path": path})
                embeddings_df = pd.DataFrame(embeddings_for_df)
                embeddings_df.to_csv(cache_destination, index=False)

        return embeddings_by_partition

    @torch.no_grad()
    def _get_embeddings(self, dataset: data_utils.Dataset) -> tuple[torch.Tensor, list[str]]:
        """Returns embeddings and path to the image files"""
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        embeddings = []
        for batch in tqdm.tqdm(dataloader):
            embeddings.append(self.embedding_model(batch[0].to(self.device)))

        return torch.cat(embeddings, dim=0), dataset.samples

    def _self_dedublication(self, embeddings: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        """Removes images that are too similar to each other"""

        embeddings = embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        D, I = index.search(embeddings, self.k_nearest_neighbors + 1)

        G = nx.Graph()
        for idx, distances in enumerate(D):
            for neighbor_idx, distance in zip(I[idx], distances):
                if distance < self.similarity_threshold_self:
                    G.add_edge(idx, neighbor_idx)

        connected_components = nx.connected_components(G)
        representatives = self._gather_representative_idxs(connected_components, self.number_of_representatives)
        deduplicated_embeddings = embeddings[representatives]

        return representatives, torch.Tensor(deduplicated_embeddings)

    @staticmethod
    def _gather_representative_idxs(connected_components: list[set[int]], num_representatives: int) -> list[int]:
        representatives = []
        for component in connected_components:
            representatives.extend(list(component)[:num_representatives])
        return representatives

    def _relative_dedublication(
        self,
        source_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        """Removes images from source that are too similar to images from reference"""
        source_embeddings = source_embeddings.cpu().numpy()
        reference_embeddings = reference_embeddings.cpu().numpy()
        faiss.normalize_L2(source_embeddings)
        faiss.normalize_L2(reference_embeddings)

        combined_embeddings = np.vstack((source_embeddings, reference_embeddings))
        dimension = combined_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

        index.add(combined_embeddings)

        D, I = index.search(source_embeddings, self.k_nearest_neighbors + 1)

        G = nx.Graph()
        num_source_images = source_embeddings.shape[0]
        for idx, similarities in enumerate(D):
            for neighbor_idx, similarity in zip(I[idx], similarities):
                if similarity < self.similarity_threshold_relative and neighbor_idx >= num_source_images:
                    G.add_edge(idx, neighbor_idx)

        to_discard = set()
        for component in nx.connected_components(G):
            if any(idx >= num_source_images for idx in component):
                to_discard.update(component)

        deduplicated_indices = [i for i in range(num_source_images) if i not in to_discard]
        relatively_deduplicated_embeddings = source_embeddings[deduplicated_indices]

        return deduplicated_indices, torch.Tensor(relatively_deduplicated_embeddings)

    def _self_supervised_sample_based_image_retrieval(self, embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _self_supervised_cluster_based_image_retrieval(self, embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


cur = CurationPipeline(embedding_model_path="./models/efficient_net_pretrained.ckpt")
cur.curate_patitioned_dataset(
    source="./data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25",
    destination="./data/splits/talk-to-kajo-test",
)
