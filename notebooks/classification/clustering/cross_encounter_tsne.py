from itertools import product
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gorillatracker.classification.clustering import EXT_MERGED_DF
from gorillatracker.classification.thesis_latex import latexify

latexify()

metric = "euclidean"
df = EXT_MERGED_DF
spac = df[(df["dataset"] == "SPAC+min3") & (df["model"] == "ViT-Finetuned")]
bristol = df[(df["dataset"] == "Bristol") & (df["model"] == "ViT-Finetuned")]


def run_tsne_and_plot(ax, df, title, metric="euclidean"):
    # Extract embeddings and labels
    embeddings = np.vstack(df["embedding"].values)
    labels = df["label"].values

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=44, metric=metric)
    tsne_results = tsne.fit_transform(embeddings)

    # Create a dataframe for the results
    tsne_df = pd.DataFrame(tsne_results, columns=["tsne_1", "tsne_2"])
    tsne_df["label"] = labels

    # Set up distinct color palette and markers
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", 10)
    markers = ["o", "s", "D", "P", "^", "v", ">", "<", "H", "*", "X", "d"]

    # Create the product of colors and markers
    symbols = list(product(markers, palette))

    # Check if we have enough unique color/shape combinations
    if len(unique_labels) > len(symbols):
        raise ValueError(
            f"Not enough unique color/shape combinations for {len(unique_labels)} labels. "
            f"We have {len(symbols)} combinations."
        )

    # Plot each label with a different color/marker combination
    for i, label in enumerate(unique_labels):
        marker, color = symbols[i]

        sns.scatterplot(
            x="tsne_1",
            y="tsne_2",
            data=tsne_df[tsne_df["label"] == label],
            marker=marker,
            color=color,
            label=label,
            s=100,
            alpha=0.85,
            ax=ax,
            legend=False,
        )

    # Remove axis labels and ticks
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    ax.set_title(title, fontsize=14)


# Assuming df, spac, and bristol are already defined as in your code
grid = (1, 2)
fig, (ax1, ax2) = plt.subplots(*grid, figsize=(16, 8))

run_tsne_and_plot(ax1, spac, "SPAC+min3", metric=metric)
run_tsne_and_plot(ax2, bristol, "Bristol", metric=metric)
plt.tight_layout()

name = f"results/cross_encounter_tsne_{metric}.pdf"
path = f"/workspaces/gorillatracker/{name}"
plt.savefig(path, dpi=300, bbox_inches="tight")

print(
    r"""
\begin{figure}[htb]
    \includegraphics[width = 1.0\textwidth]{"""
    + name
    + r"""}
    \caption{"""
    + "TSNE embeddings of SPAC+min3 and Bristol datasets using ViT-Finetuned model "
    + ("(Euclidean Distance)" if metric == "euclidean" else "(Cosine Similarity)")
    + r"""}
    \label{fig:objective-functions-graphs-"""
    + metric
    + r"""}
\end{figure}
"""
)
