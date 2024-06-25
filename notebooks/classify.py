# A tool to make classification easy.
"""
1. Read a dataset into (partition, id, label, image(path), metadata) format.
2. Compute embeddings via a model, and save them (partition, id, label, image(path), embedding, metadata).
3. Analyze the clusters and visualize them. (TSNE, PCA, UMAP)
4. Implement multiple methods to classify the embeddings.
    a. KNN
"""

from gorillatracker.scripts.visualize_embeddings import EmbeddingProjector
from io import BytesIO
import numpy as np
from bokeh.io import show, output_notebook, reset_output
import base64


def embedding_projector(df):
    output_notebook()
    embeddings = np.stack(df["embedding"])

    images = []
    for image in df["input"]:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_byte = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images.append(image_byte)

    ep = EmbeddingProjector()
    low_dim_embeddings = ep.reduce_dimensions(embeddings)
    fig = ep.plot_clusters(
        low_dim_embeddings, df["label"], df["label_string"], images, title="Embedding Projector", figsize=(12, 10)
    )
    show(fig)
    reset_output()
