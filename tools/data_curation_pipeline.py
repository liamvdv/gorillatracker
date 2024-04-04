import pathlib
import torch
import torch.utils.data as data_utils
import tqdm
import faiss
import numpy as np
import networkx as nx
import logging
import pandas as pd
import hashlib
import gorillatracker.model as model
import gorillatracker.datasets.cxl as cxl

from typing import Literal

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

    def __init__(self, embedding_model_path: str, embedding_model=model.EfficientNetV2Wrapper):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_class = cxl.CXLDataset

        # dataloader settings
        self.batch_size = 32
        self.num_workers = 8

        # curation settings
        self.cache_dir = pathlib.Path("./data/embeddings")

        self.k_nearest_neighbors = 8  # number of nearest neighbors to consider for clustering
        self.similarity_threshold_self = (
            0.15  # similarity threshold for self dedublication. Higher values will result in more images being removed
        )
        self.number_of_representatives = 3  # number of representatives to keep from each cluster. Higher values will result in less images being removed
        self.similarity_threshold_relative = 0.17  # similarity threshold for relative dedublication.  Higher values will result in more images being removed

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
        partitions = ["train", "val", "test"]

        embeddings_df = self._get_embeddings_by_partition(source=source, partitions=partitions)
        dedublicated_embeddings_df = self._self_dedublication_by_partition(embeddings_df, partitions)
        relative_deduplicated_embeddings_df = self._relative_dedublication(
            dedublicated_embeddings_df, source="train", reference="test"
        )

        print(embeddings_df["partition"].value_counts())
        print(dedublicated_embeddings_df["partition"].value_counts())
        print(relative_deduplicated_embeddings_df["partition"].value_counts())

        dedublicated_embeddings_df.to_pickle(destination + "/deduplicated_embeddings.pkl")
        relative_deduplicated_embeddings_df.to_pickle(destination + "/relative_embeddings.pkl")

    def _get_embeddings_by_partition(
        self,
        source: str,
        partitions: list[str],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Retrieves the embeddigns for all partitions given from the source path

        Returns:
            DataFrame in the format {"embedding": torch.Tensor, "path": str}
        """
        cache_destination = (
            self.cache_dir
            / f"{hashlib.sha256((source + str(self.embedding_model_path) + str(partitions)).encode()).hexdigest()}.pkl"
        )
        if use_cache and cache_destination.exists():
            return pd.read_pickle(cache_destination)

        embeddings_df_list = []
        for partition in partitions:
            logger.info("Gathering embeddings for partion: %s", partition)

            dataset = self.dataset_class(
                data_dir=source, partition=partition, transform=self.dataset_class.get_transforms()
            )
            embeddings, paths = self._get_embeddings(dataset)

            for embedding, path in zip(embeddings, paths):
                embeddings_df_list.append({"partition": partition, "embedding": embedding.numpy(), "path": str(path)})

        embeddings_df = pd.DataFrame(embeddings_df_list)
        logger.info(f"Loaded embeddings for {len(embeddings_df)} images.")
        embeddings_df.to_pickle(cache_destination)

        return embeddings_df

    @torch.no_grad()
    def _get_embeddings(self, dataset: data_utils.Dataset) -> tuple[torch.Tensor, list[str]]:
        """Returns embeddings and path to the image files"""
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        embeddings = []
        for batch in tqdm.tqdm(dataloader):
            embeddings.append(self.embedding_model(batch[0].to(self.device)).cpu())

        return torch.cat(embeddings, dim=0), dataset.samples

    def _self_dedublication_by_partition(self, embedding_df: pd.DataFrame, partitions: Literal[str]) -> pd.DataFrame:
        """Removes images that are too similar to each other within the same partition"""
        deduplicated_embeddings_df = []
        for partition in partitions:
            logger.info("Deduplicating partition: %s", partition)
            partition_df = embedding_df[embedding_df["partition"] == partition]
            representatives_idx, _ = self._self_dedublication(np.vstack(partition_df["embedding"].to_numpy()))
            deduplicated_embeddings_df.append(partition_df.iloc[representatives_idx])

        return pd.concat(deduplicated_embeddings_df)

    def _self_dedublication(self, embeddings: np.ndarray) -> tuple[list[int], list[torch.Tensor]]:
        """Removes images that are too similar to each other"""
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        D, I = index.search(embeddings, self.k_nearest_neighbors + 1)

        G = nx.Graph()
        for idx, distances in enumerate(tqdm.tqdm(D)):
            for neighbor_idx, distance in zip(I[idx], distances):
                if distance < self.similarity_threshold_self:
                    G.add_edge(idx, neighbor_idx)

        connected_components = nx.connected_components(G)
        representatives = self._gather_representative_idxs(connected_components, self.number_of_representatives)
        deduplicated_embeddings = embeddings[representatives]

        return representatives, deduplicated_embeddings

    @staticmethod
    def _gather_representative_idxs(connected_components: list[set[int]], num_representatives: int) -> list[int]:
        representatives = []
        for component in connected_components:
            representatives.extend(list(component)[:num_representatives])
        return representatives

    def _relative_dedublication(
        self,
        embedding_df: pd.DataFrame,
        source: Literal["train", "val", "test"],
        reference: Literal["train", "val", "test"],
    ) -> pd.DataFrame:
        """Removes images from source that are too similar to images from reference"""
        source_df = embedding_df[embedding_df["partition"] == source]
        reference_df = embedding_df[embedding_df["partition"] == reference]
        source_embeddings = np.vstack(source_df["embedding"].to_list())
        reference_embeddings = np.vstack(reference_df["embedding"].to_list())
        faiss.normalize_L2(source_embeddings)
        faiss.normalize_L2(reference_embeddings)

        combined_embeddings = np.vstack((source_embeddings, reference_embeddings))
        dimension = combined_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

        index.add(combined_embeddings)

        D, I = index.search(source_embeddings, self.k_nearest_neighbors + 1)

        G = nx.Graph()
        num_source_images = source_embeddings.shape[0]
        for idx, similarities in enumerate(tqdm.tqdm(D)):
            for neighbor_idx, similarity in zip(I[idx], similarities):
                if similarity < self.similarity_threshold_relative and neighbor_idx >= num_source_images:
                    G.add_edge(idx, neighbor_idx)

        to_discard = set()
        for component in nx.connected_components(G):
            if any(idx >= num_source_images for idx in component):
                to_discard.update(component)

        deduplicated_indices = [i for i in range(num_source_images) if i not in to_discard]

        return pd.concat([source_df.iloc[deduplicated_indices], embedding_df[embedding_df["partition"] != source]])


class SSLCurationPipeline:
    def __init__(self) -> None:
        pass

    def load_tracking_data(self, source: str) -> pd.DataFrame:
        pass


if __name__ == "__main__":
    # cur = CurationPipeline(embedding_model_path="./models/efficient_net_pretrained.ckpt")
    # cur.curate_patitioned_dataset(
    #     source="./data/splits/derived_data-spac_gorillas_converted_labels_cropped_faces-train-openset-reid-val-10-test-10-mintraincount-3-seed-42-train-70-val-15-test-15",
    #     # source="./data/splits/ground_truth-cxl-face_image_detection_90degree-anno-seed-42-train-70-val-15-test-15",
    #     destination="./data/embeddings/talk-to-kajo-test",
    # )
    ssl_cur = SSLCurationPipeline()
    ssl_cur.load_tracking_data(source="./data/derived_data/tracking_data.csv")
