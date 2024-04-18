import base64
from io import BytesIO

import colorcet as cc
import numpy as np
import numpy.typing as npt
import pandas as pd
import umap.umap_ as umap
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding


class EmbeddingProjector:
    def __init__(self) -> None:
        self.algorithms = {
            "tsne": TSNE(n_components=2),
            "isomap": Isomap(n_components=2),
            "lle": LocallyLinearEmbedding(n_components=2),
            "mds": MDS(n_components=2),
            "spectral": SpectralEmbedding(n_components=2),
            "pca": PCA(n_components=2),
            "umap": umap.UMAP(),
        }

    def reduce_dimensions(self, embeddings: npt.NDArray[np.float_], method: str = "tsne") -> npt.NDArray[np.float_]:
        assert len(embeddings) > 2
        algorithm = TSNE(n_components=2, perplexity=1)
        if len(embeddings) > 30:
            algorithm = self.algorithms.get(method, TSNE(n_components=2))
        return algorithm.fit_transform(embeddings)

    def plot_clusters(
        self,
        low_dim_embeddings: npt.NDArray[np.float_],
        labels: pd.Series,
        og_labels: pd.Series,
        images: list[
            str
        ],  # base64.b64encode(buffer.getvalue()).decode("utf-8") --> buffer is a JPEG image inside of a BytesIO object
        title: str = "Embedding Projector",
        figsize: tuple[int, int] = (12, 10),
    ) -> figure:
        """
        after calling this use:

        if in notebook:
        show(fig)

        if in script:
        output_file(filename="embedding.html")
        save(fig)
        """
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

        return fig


def visualize_embeddings(
    df: pd.DataFrame,
    label_column: str = "label",
    label_string_column: str = "label_string",
    embedding_column: str = "embedding",
    image_column: str = "input",
    figsize: tuple[int, int] = (12, 10),
    dimension_reduce_method: str = "tsne",
) -> None:
    embeddings = df[embedding_column].to_numpy()
    embeddings = np.stack(embeddings)

    images = []
    for image in df[image_column]:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_byte = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images.append(image_byte)

    ep = EmbeddingProjector()
    low_dim_embeddings = ep.reduce_dimensions(embeddings, method=dimension_reduce_method)
    ep.plot_clusters(
        low_dim_embeddings, df[label_column], df[label_string_column], images, title="Embeddings", figsize=(12, 10)
    )


if __name__ == "__main__":
    ec = EmbeddingProjector()
