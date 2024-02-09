import faiss
import torch
import gorillatracker.utils.embedding_generator as eg
import numpy as np

from typing import List


class LabelGatherer:
    def __init__(self) -> None:
        self.embeddings_df = eg.read_embeddings_from_disk("./data/embeddings/swin_v2_cxl_open.pkl")
        self.index = self._get_faiss_index(self.embeddings_df["embedding"].tolist())

    def _get_faiss_index(self, embeddings: list[torch.Tensor]) -> faiss.IndexFlatL2:
        """Returns the labels of the embeddings"""
        embeddings = np.array(embeddings)  # type: ignore
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index

    def _get_label_for_embedding(self, embedding: torch.Tensor, k: int = 1) -> List[str]:
        """Returns the label for the given embedding"""
        embedding = embedding.unsqueeze(0).numpy()
        faiss.normalize_L2(embedding)
        _, I = self.index.search(embedding, k)
        return self.embeddings_df.iloc[I[0][0]]["label_string"]

    def get_labels_for_embeddings(self, embeddings: dict) -> set[str]:
        """Returns the labels for the given embeddings"""
        labels = []
        for embedding_value in embeddings.values():
            labels.append(self._get_label_for_embedding(embedding_value["embedding"]))
        return set(labels)


labl = LabelGatherer()
print(labl.get_labels_for_embeddings({"1": {"embedding": torch.rand(256)}}))
