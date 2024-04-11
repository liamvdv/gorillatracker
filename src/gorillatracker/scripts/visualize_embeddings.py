from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import numpy as np
from io import BytesIO
import base64
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
import colorcet as cc
import pandas as pd

class EmbeddingProjector:
    def __init__(self):
        self.algorithms = {
            "tsne": TSNE(n_components=2),
            "isomap": Isomap(n_components=2),
            "lle": LocallyLinearEmbedding(n_components=2),
            "mds": MDS(n_components=2),
            "spectral": SpectralEmbedding(n_components=2),
            "pca": PCA(n_components=2),
            "umap": umap.UMAP(),
        }

    def reduce_dimensions(self, embeddings, method="tsne"):
        # handle --fast_dev_run where there is a reduced number of embeddings
        assert len(embeddings) > 2
        algorithm = TSNE(n_components=2, perplexity=1)
        if len(embeddings) > 30:
            algorithm = self.algorithms.get(method, TSNE(n_components=2))
        return algorithm.fit_transform(embeddings)

    def plot_clusters(
        self, low_dim_embeddings, labels, og_labels, images, title="Embedding Projector", figsize=(12, 10)
    ):
        color_names = cc.glasbey
        color_lst = [color_names[label * 2] for label in labels]
        data = {
            "x": low_dim_embeddings[:, 0],
            "y": low_dim_embeddings[:, 1],
            "color": color_lst,
            "class": og_labels,
            "image": images,
        }

        fig = figure(tools="pan, wheel_zoom, box_zoom, reset")
        fig.scatter(
            x="x",
            y="y",
            size=12,
            fill_color="color",
            line_color="black",
            source=ColumnDataSource(data=data),
            legend_field="class",
        )

        hover = HoverTool(tooltips='<img src="data:image/jpeg;base64,@image" width="128" height="128">')
        fig.add_tools(hover)

        output_file(filename = "embedding.html")
        save(fig)


def visualize_embeddings(df: pd.DataFrame, label_column: str = "label", label_string_column: str = "label_string",
                        embedding_column: str = "embedding", image_column: str = "input", figsize: tuple = (12, 10)):
    embeddings = df[embedding_column].to_numpy()
    embeddings = np.stack(embeddings)

    images = []
    for image in df[image_column]:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_byte = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images.append(image_byte)

    ep = EmbeddingProjector()
    low_dim_embeddings = ep.reduce_dimensions(embeddings, method="tsne")
    ep.plot_clusters(
        low_dim_embeddings, df[label_column], df[label_string_column], images, title="Embeddings", figsize=(12, 10)
    )