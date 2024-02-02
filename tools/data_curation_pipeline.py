import pathlib
import torch
import torch.utils.data as data_utils
import tqdm
import faiss
import numpy as np
import networkx as nx
import logging
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
        self.reduction_factor = 64  # reduce the number of images by this factor
        self.similarity_threshold_self = 0.6  # similarity threshold for self dedublication
        self.similarity_threshold_relative = 0.45  # similarity threshold for relative dedublication

        # setup embedding model
        self.embedding_model_path = embedding_model_path
        self.embedding_model = embedding_model(
            model_name_or_path=self.embedding_model_path, **embedding_model_settings
        ).to(self.device)
        self.embedding_model.load_state_dict(torch.load(self.embedding_model_path)["state_dict"])
        self.embedding_model.eval()

        logger.info("CurationPipeline successfully initialized!")

    def curate_patitioned_dataset(self, source: str, destination: str) -> None:
        source = pathlib.Path(source)
        destination = pathlib.Path(destination)
        logger.info("Curating dataset from source: %s to destination: %s", source, destination)
        partitions = ["test"]  # TODO: replace with ["train", "val", "test"]

        for partition in partitions:
            logger.info("Curating partition: %s", partition)
            dataset = self.dataset_class(
                data_dir=source, partition=partition, transform=self.dataset_class.get_transforms()
            )
            self._curate_dataset(dataset, destination / partition)

    def _curate_dataset(self, dataset: data_utils.Dataset, destination: pathlib.Path) -> None:
        embeddings, paths = self._get_embeddings(dataset)
        print(embeddings.shape)

        # raise NotImplementedError()

    @torch.no_grad()
    def _get_embeddings(self, dataset: data_utils.Dataset) -> tuple[torch.Tensor, list[str]]:
        """Returns embeddings and path to the image files"""
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        embeddings = []
        for batch in tqdm.tqdm(dataloader):
            embeddings.append(self.embedding_model(batch[0].to(self.device)))

        return torch.cat(embeddings, dim=0), dataset.samples[1]

    def _self_dedublication(self, embeddings: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        """Removes images that are too similar to each other"""

        embeddings = embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        D, I = index.search(embeddings, self.reduction_factor + 1)
        distance_threshold = 2 * (1 - 0.6)

        G = nx.Graph()
        for idx, distances in enumerate(D):
            for neighbor_idx, distance in zip(I[idx], distances):
                if distance < distance_threshold:
                    G.add_edge(idx, neighbor_idx)

        connected_components = nx.connected_components(G)
        representatives = [list(component)[0] for component in connected_components]
        deduplicated_embeddings = embeddings[representatives]

        return representatives, torch.Tensor(deduplicated_embeddings)

    def _relative_dedublication(
        self,
        source_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        """Removes images from source that are too similar to images from reference"""
        faiss.normalize_L2(source_embeddings)
        faiss.normalize_L2(reference_embeddings)

        combined_embeddings = np.vstack((source_embeddings, reference_embeddings))
        dimension = combined_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        index.add(combined_embeddings)

        D, I = index.search(source_embeddings, self.reduction_factor + 1)

        similarity_threshold = 0.45

        G = nx.Graph()
        num_source_images = source_embeddings.shape[0]
        for idx, similarities in enumerate(D):
            for neighbor_idx, similarity in zip(I[idx], similarities):
                if similarity > similarity_threshold and neighbor_idx >= num_source_images:
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
