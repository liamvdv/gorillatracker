import faiss
import torch
import gorillatracker.utils.embedding_generator as eg
import numpy as np
import pandas as pd
import pathlib
import hashlib

from typing import List, Optional


class LabelGatherer:
    def __init__(
        self,
        model_from_run: str,
        use_cache: bool = True,
        data_dir: Optional[str] = None,
        dataset_class: Optional[str] = None,
    ) -> None:
        self.cache_destination = "./data/embeddings"
        self.embeddings_df = self._get_embeddings(model_from_run, use_cache, data_dir, dataset_class)
        self.index = self._get_faiss_index(self.embeddings_df["embedding"].tolist())

    def _get_embeddings(
        self,
        model_from_run: str,
        use_cache: bool = True,
        data_dir: Optional[str] = None,
        dataset_class: Optional[str] = None,
    ) -> pd.DataFrame:
        combined_name_bytes = (model_from_run + str(data_dir)).encode()
        embeddings_on_disk_path = pathlib.Path(self.cache_destination) / (
            hashlib.sha256(combined_name_bytes).hexdigest() + ".pkl"
        )
        print(embeddings_on_disk_path.exists())
        if use_cache and embeddings_on_disk_path.exists():
            return eg.read_embeddings_from_disk(str(embeddings_on_disk_path))

        return eg.generate_embeddings_from_run(
            run_url=model_from_run,
            outpath=str(embeddings_on_disk_path if use_cache else "-"),
            data_dir=data_dir,
            dataset_class=dataset_class,
        )

    def _get_faiss_index(self, embeddings: list[torch.Tensor]) -> faiss.IndexFlatL2:
        embeddings = np.array(embeddings)  # type: ignore
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index

    def get_label_for_embedding(self, embedding: torch.Tensor, k: int = 1) -> List[str]:
        embedding = embedding.unsqueeze(0).numpy()
        faiss.normalize_L2(embedding)
        _, I = self.index.search(embedding, k)
        return self.embeddings_df.iloc[I[0][0]]["label_string"]


if __name__ == "__main__":
    labl = LabelGatherer(
        model_from_run="https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/69ok0oyl"
    )

