import faiss
import gorillatracker.datasets.cxl as cxl
import numpy as np


class LabelGatherer:
    def __init__(self) -> None:
        self.cxl = cxl.CXLDataset(
            "data/DO-NOT-USE-PING-BEN-IF-IN-DOUBT-joined_splits/cxl_face-openset=True_0",
            "train",
            CXLDataset.get_transforms(),
        )

    def get_labels_for_embeddings(embeddings: list):
        """Returns the labels of the embeddings"""
        embeddings = np.array(embeddings)
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        D, I = index.search(embeddings, 1)

        return I.flatten()
