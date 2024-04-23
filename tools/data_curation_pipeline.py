import hashlib
import json
import logging
import pathlib
import shutil
from typing import Literal, Tuple

import faiss
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import tqdm

import gorillatracker.datasets.cxl as cxl
import gorillatracker.datasets.spac_videos as spac_videos
import gorillatracker.model as model

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
    "stepwise_schedule": False,
    "lr_interval": 1.0,
    "l2_alpha": 0.1,
    "l2_beta": 0.01,
    "path_to_pretrained_weights": "./models/swin_base_untrained.ckpt",
    "margin": 1.0,
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
            0.08  # similarity threshold for self dedublication. Higher values will result in more images being removed
        )
        self.number_of_representatives = 5  # number of representatives to keep from each cluster. Higher values will result in less images being removed
        self.similarity_threshold_relative = 0.12  # similarity threshold for relative dedublication.  Higher values will result in more images being removed

        # setup embedding model
        self.embedding_model_path = embedding_model_path
        self.embedding_model = embedding_model(
            model_name_or_path=self.embedding_model_path, **embedding_model_settings
        ).to(self.device)
        state_dict = torch.load(self.embedding_model_path)["state_dict"]
        self.embedding_model.load_state_dict(state_dict)
        self.embedding_model.eval()

        logger.info("CurationPipeline successfully initialized!")

    def curate_patitioned_dataset(self, source: str, destination: str):
        logger.info("Curating dataset from source: %s to destination: %s", source, destination)
        partitions = ["train", "val", "test"]

        embeddings_df = self._get_embeddings_by_partition(source=source, partitions=partitions)
        curated_df = self._curate_dataframe(embeddings_df, source, destination)

        for _, row in tqdm.tqdm(curated_df.iterrows(), total=len(curated_df)):
            source_path = pathlib.Path(row["path"])
            relative_path = source_path.relative_to(pathlib.Path(source))
            destination_path = pathlib.Path(destination) / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, destination_path)

    def _curate_dataframe(self, dataframe: pd.DataFrame, source: str, destination: str) -> pd.DataFrame:
        embeddings_df = dataframe
        dedublicated_embeddings_df = self._self_dedublication_by_partition(embeddings_df, ["train"])
        relative_deduplicated_embeddings_df = self._relative_dedublication(
            dedublicated_embeddings_df, source="train", reference="test"
        )

        print(embeddings_df["partition"].value_counts())
        print(dedublicated_embeddings_df["partition"].value_counts())
        print(relative_deduplicated_embeddings_df["partition"].value_counts())

        return relative_deduplicated_embeddings_df

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
            embeddings, paths, labels = self._get_embeddings(dataset)

            for embedding, path, labels in zip(embeddings, paths, labels):
                embeddings_df_list.append(
                    {"partition": partition, "embedding": embedding.numpy(), "path": str(path), "label": labels}
                )

        embeddings_df = pd.DataFrame(embeddings_df_list)
        logger.info(f"Loaded embeddings for {len(embeddings_df)} images.")
        embeddings_df.to_pickle(cache_destination)

        return embeddings_df

    @torch.no_grad()
    def _get_embeddings(self, dataset: data_utils.Dataset) -> tuple[torch.Tensor, list[str], list[str]]:
        """Returns embeddings and path to the image files"""
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        embeddings = []
        for batch in tqdm.tqdm(dataloader):
            embeddings.append(self.embedding_model(batch[1].to(self.device)).cpu())
        paths, labels = zip(*dataset.samples)
        return torch.cat(embeddings, dim=0), paths, labels

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
        
        deduplicated_indices = self._find_deduplicated_indices(source_embeddings, D, I)
        
        return pd.concat([source_df.iloc[deduplicated_indices], embedding_df[embedding_df["partition"] != source]])
    
    def _find_deduplicated_indices(self, source_embeddings: np.ndarray, D, I) -> set:
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
                
        return deduplicated_indices
        


class SSLCurationPipeline(CurationPipeline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_class = spac_videos.SPACVideosDataset
        self.min_negative_count = 1

    def curate_patitioned_dataset(self, source: str, destination: str):
        logger.info("Curating dataset from source: %s to destination: %s", source, destination)
        partitions = ["train", "val", "test"]

        embeddings_df = self._get_embeddings_by_partition(source=source, partitions=partitions)
        curated_df, leftover_negatives = self._curate_dataframe(embeddings_df, source, destination)

        print(curated_df["partition"].value_counts())

        pathlib.Path(destination + "/train").mkdir(parents=True, exist_ok=True)
        json.dump(leftover_negatives, open(destination + "/train/negatives.json", "w"))

        for _, row in tqdm.tqdm(curated_df.iterrows(), total=len(curated_df)):
            source_path = pathlib.Path(row["path"])
            relative_path = source_path.relative_to(pathlib.Path(source))
            destination_path = pathlib.Path(destination) / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, destination_path)

    def _curate_dataframe(self, dataframe: pd.DataFrame, source: str, destination: str) -> Tuple[pd.DataFrame, dict]:
        """
        Assumes directory structure:
            data_dir/
                train/
                    video_data
                    negatives.json
                val/
                    cxl_data
                test/
                    cxl_data
        """
        negatives = json.load(open(source + "/train/negatives.json"))

        curated_df = super()._curate_dataframe(dataframe, source, destination)

        ids_to_keep = curated_df["label"]
        ids_to_keep_in_train = curated_df[curated_df["partition"] == "train"]["label"]
        additional_ids_to_keep = []
        negative_counter_for_ids_to_keep = {id: 0 for id in ids_to_keep_in_train}

        # record how many negatives each individual has in the dataset
        for id in ids_to_keep_in_train:
            negative_ids = negatives[id]
            for negative_id in negative_ids:
                if negative_id in negative_counter_for_ids_to_keep:
                    negative_counter_for_ids_to_keep[negative_id] += 1

        # expand the dataset with negatives so search id to keep has at least n negatives
        for id, count in negative_counter_for_ids_to_keep.items():
            if count < self.min_negative_count:
                additional_ids_to_keep.extend(negatives[id][: min(self.min_negative_count - count, len(negatives[id]))])

        leftover_negatives = {id: negatives[id] for id in ids_to_keep_in_train}
        ids_to_keep = list(ids_to_keep) + additional_ids_to_keep

        return dataframe[dataframe["label"].isin(ids_to_keep)], leftover_negatives


if __name__ == "__main__":
    # cur = CurationPipeline(embedding_model_path="./models/efficient_net_pretrained.ckpt")
    # cur.curate_patitioned_dataset(
    #     source="./data/splits/derived_data-spac_gorillas_converted_labels_cropped_faces-train-openset-reid-val-10-test-10-mintraincount-3-seed-42-train-70-val-15-test-15",
    #     # source="./data/splits/ground_truth-cxl-face_image_detection_90degree-anno-seed-42-train-70-val-15-test-15",
    #     destination="./data/embeddings/talk-to-kajo-test",
    # )

    # samples = spac_videos.get_samples_video(
    #     pathlib.Path("./data/derived_data/spac_gorillas_converted_labels_cropped_faces/train")
    # )

    ssl_cur = SSLCurationPipeline(embedding_model_path="./models/efficient_net_pretrained.ckpt")
    ssl_cur.curate_patitioned_dataset(
        source="./data/derived_data/spac_gorillas_converted_labels_cropped_faces",
        destination="./data/embeddings/talk-to-kajo-test",
    )
