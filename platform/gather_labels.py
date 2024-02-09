import faiss
import torch
import gorillatracker.utils.embedding_generator as eg
import numpy as np


class LabelGatherer:
    def __init__(self) -> None:
        self.embeddings_df = eg.generate_embeddings_from_run(
            run_url="https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/o5vyeckp?workspace=user-kajo-hpi",
            outpath="./data/embeddings/swin_v2_cxl_open.pkl",
        )
        self.index = self._get_faiss_index(self.embeddings_df["embedding"].tolist())

    def _get_faiss_index(self, embeddings: list[torch.Tensor]) -> faiss.IndexFlatL2:
        """Returns the labels of the embeddings"""
        embeddings = np.array(embeddings)  # type: ignore
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index

    def _get_label_for_embedding(self, embedding: torch.Tensor) -> str:
        """Returns the label for the given embedding"""
        faiss.normalize_L2(embedding)
        _, I = self.index.search(embedding.numpy(), 1)
        return self.embeddings_df.iloc[I[0][0]]["label_string"]

    def get_labels_for_embeddings(self, embeddings: dict) -> set[str]:
        """Returns the labels for the given embeddings"""
        labels = []
        for embedding_value in embeddings.values():
            labels.append(self._get_label_for_embedding(embedding_value["embedding"]))
        return set(labels)


labl = LabelGatherer()
