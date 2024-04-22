from gorillatracker.utils.embedding_generator import generate_embeddings_from_run, read_embeddings_from_disk
from gorillatracker.scripts.visualize_embeddings import EmbeddingProjector
import numpy as np
import pandas as pd
from gorillatracker.clustering.folds import DistinctClassFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    average_precision_score,
    pairwise_distances,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
import tqdm
import matplotlib.pyplot as plt
from gorillatracker.metrics import knn

# Multi-Label Model Evaluation
# https://www.kaggle.com/code/kmkarakaya/multi-label-model-evaluation


def norm_label_distribution(df, label_column, seed=42):
    at_least = 2
    df = df.groupby(label_column).filter(lambda x: len(x) >= at_least).reset_index(drop=True)
    distribution = df[label_column].value_counts()
    is_biased = (distribution.max() - distribution.min()) / len(df[label_column]) > 0.03
    if is_biased:

        def downsampler(x):
            down_lo = distribution.min()
            down_hi = distribution.min() * 2
            n = len(x)
            sample_n = max(down_lo, min(n, down_hi))
            return x.sample(sample_n, random_state=seed)

        df = df.groupby(label_column).apply(downsampler).reset_index(drop=True)
        distribution = df[label_column].value_counts()
        assert (distribution.max() - distribution.min()) / len(
            df[label_column]
        ) <= 0.03, f"Labels are not equally distributed {distribution}, pass normalize_label_distribution=True to normalize. Note that this will remove some samples."
        print(distribution)
    return df


"""
Use sth. other than grid search:
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
"""


def find_threshold(
    df: pd.DataFrame,
    label_column: str,
    grid_start: float,
    grid_end: float,
    grid_num: float,
    unique_percentage=0.2,
    seed=42,
    normalize_label_distribution: bool = False,
):
    """You are responsible for normalizing the label distribution."""
    if normalize_label_distribution:
        df = norm_label_distribution(df, label_column, seed)

    # Thresholds for grid search
    thresholds = np.linspace(grid_start, grid_end, grid_num)

    # For storing results
    results = []
    kf = DistinctClassFold(n_buckets=5, shuffle=True, random_state=seed)

    # TODO(liamvdv): handle no sufficient match (singletons cannot have match)
    new_label = "new"
    for dataset_df, query_df in kf.split(
        df,
        label_column=label_column,
        label_mask=new_label,
        unique_percentage=unique_percentage,
        normalize_label_distribution=normalize_label_distribution,
    ):
        # Compute pairwise distances between query and training embeddings
        query = np.stack(query_df["embedding"].values)
        dataset = np.stack(dataset_df["embedding"].values)
        distances = pairwise_distances(query, dataset)
        closest_indices = np.argmin(distances, axis=1)
        perc_of_new_label = query_df[label_column].value_counts().get(new_label, 0) / len(query_df[label_column])

        # Now you can get the predicted label based on the closest train embedding
        predicted_labels = dataset_df.iloc[closest_indices][label_column].values
        actual_labels = query_df[label_column].values

        # Binarize the labels for mAP calculation
        combined_labels = np.concatenate([actual_labels, [new_label]])
        unique_labels = np.unique(combined_labels)
        # NOTE(liamvdv): creates a [len(actual_labels), len(unique_labels)] classification matrix
        #                Values are 0 or 1, single 1 rest 0 per row.
        # [n_samples, n_classes]
        ground_truth_matrix = label_binarize(actual_labels, classes=unique_labels)

        for threshold in tqdm.tqdm(thresholds, "Grid Search Thresholds"):
            thresholded_labels = [
                new_label if distances[i][closest_indices[i]] > threshold else predicted_label
                for i, predicted_label in enumerate(predicted_labels)
            ]
            # Binarize the labels for mAP calculation
            prediction_matrix = label_binarize(thresholded_labels, classes=unique_labels)

            # Calculate mAP for all labels
            # map_score_all = average_precision_score(ground_truth_matrix, prediction_matrix, average="macro")

            # Calculate mAP for the 'new' label
            assert new_label in unique_labels
            new_index = unique_labels.tolist().index(new_label)

            # Calculate the samples mask (indices) that are 'new' in ground_truth
            np.set_printoptions(threshold=np.inf)
            # alias is_new_row_mask

            # [:,<idx>] only selects a given column, here the new_label column
            # We'll look at this as a 'new' or not new problem
            # map_score_new = average_precision_score(ground_truth_matrix[:, new_index], prediction_matrix[:, new_index], average="macro")

            # Calculate mAP for all other non-new labels
            non_new_indices = [i for i, label in enumerate(unique_labels) if label != new_label]

            # not_new_row_mask = np.where(ground_truth_matrix[:, new_index] != 1)[0]
            # assert ground_truth_matrix[:,non_new_indices].sum(axis=1).min() == 1, "There must be exactly 1 classification per row."

            for i, label in enumerate(unique_labels):
                assert np.any(
                    ground_truth_matrix[:, i] == 1
                ), f"No instance of label {i} '{label}' is not present in the ground truth matrix."

            # print(ground_truth_matrix[not_new_row_mask][:,non_new_indices])
            # map_score_non_new = average_precision_score(
            #     ground_truth_matrix[:, non_new_indices],
            #     prediction_matrix[:, non_new_indices],
            #     average="macro"
            # )
            metrics = dict()
            for slug, Y_true, Y_pred in [
                ("all", ground_truth_matrix, prediction_matrix),
                ("new", ground_truth_matrix[:, new_index], prediction_matrix[:, new_index]),
                ("non_new", ground_truth_matrix[:, non_new_indices], prediction_matrix[:, non_new_indices]),
            ]:
                metrics.update(
                    {
                        f"{slug}/average_precision_score": average_precision_score(Y_true, Y_pred, average="macro"),
                        f"{slug}/roc_auc_score": roc_auc_score(Y_true, Y_pred, labels=unique_labels),
                        # f"{slug}/classification_report": classification_report(Y_true, Y_pred, target_names=unique_labels),
                        "percent_of_new_label": perc_of_new_label,
                    }
                )
            results.append((threshold, metrics))

            # # print(auc_score)
            # # RocCurveDisplay.from_predictions(ground_truth_matrix[:, new_index], prediction_matrix[:, new_index], name=f"{new_index} vs the rest", plot_chance_level=True).plot()
            # # plt.savefig("auc.png")
            # results.append((threshold, map_score_all, map_score_new, map_score_non_new, perc_of_new_label))

        # results = np.array(results)
        # unique_thresholds = np.unique(results[:, 0])
        # map_per_threshold = {
        #     threshold: np.mean(results[results[:, 0] == threshold], axis=0)[1:] for threshold in unique_thresholds
        # }
        aggregated = {
            threshold: {
                name: np.mean([metrics[name] for inner_threshold, metrics in results if inner_threshold == threshold])
                for name in results[0][1].keys()
            }
            for threshold in thresholds
        }
        return aggregated


def select_best_threshold(map_per_threshold: dict):
    best_threshold, best_overall_map, new_map, non_new_map, perc_of_new_label = max(
        [[label, *mAPs] for label, mAPs in map_per_threshold.items()], key=lambda x: x[1]
    )
    return best_threshold, best_overall_map, new_map, non_new_map, perc_of_new_label


def plot_map_vs_thresholds(map_per_threshold: dict, title: str, path: str):
    # Extracting the data for plotting
    thresholds = list(map_per_threshold.keys())
    mAP_overall = [values[0] for values in map_per_threshold.values()]
    mAP_new = [values[1] for values in map_per_threshold.values()]
    mAP_non_new = [values[2] for values in map_per_threshold.values()]
    perc_of_new_label = [values[3] for values in map_per_threshold.values()][0]

    # Creating the plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mAP_overall, label="Overall", marker="o")
    plt.plot(thresholds, mAP_new, label="Only New", marker="o")
    plt.plot(thresholds, mAP_non_new, label="Only Non-New", marker="o")

    # Adding title and labels
    plt.title(title + f" @ mean {perc_of_new_label * 100:.2f}% new individuals in query set")
    plt.xlabel("Threshold")
    plt.ylabel("mAP")
    plt.legend()

    # Showing the plot
    plt.grid(True)
    plt.savefig(path)


def test_find_thresholds_is_deterministic():
    # generate_embeddings_from_run("https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/cc6tiy3f/workspace", "example-embeddings.pkl")
    df = read_embeddings_from_disk("example-embeddings.pkl")
    # print(df.head())
    # Convert embeddings to numpy arrays if they are not already
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))

    results = []
    seed = 777
    for run in range(3):
        unique_percentage = 0.2
        map_per_threshold = find_threshold(
            df=df,
            label_column="label_string",
            grid_start=7.0,
            grid_end=20.0,
            grid_num=20,
            unique_percentage=unique_percentage,
            seed=seed,
            normalize_label_distribution=True,
        )
        result = select_best_threshold(map_per_threshold)
        results.append(result)

    assert all(
        [result == results[0] for result in results]
    ), "Results are not the same for different runs (not working deterministically)"


import base64
from io import BytesIO
from bokeh.io import output_file, save


def visualize_embeddings(df):
    images = []
    for image in df["input"]:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_byte = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images.append(image_byte)

    ep = EmbeddingProjector()
    low_dim_embeddings = ep.reduce_dimensions(np.stack(df["embedding"]), method="tsne")
    fig = ep.plot_clusters(
        low_dim_embeddings, df["label"], df["label_string"], images, title="Embedding Projector", figsize=(12, 10)
    )
    output_file(filename="embedding.html")
    save(fig)


if __name__ == "__main__":
    # generate_embeddings_from_run("https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/cc6tiy3f/workspace", "example-embeddings.pkl")
    df = read_embeddings_from_disk("example-embeddings.pkl")
    # print(df.head())
    # Convert embeddings to numpy arrays if they are not already
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))

    visualize_embeddings(df)
    # for i in range(10):
    #     seed = np.random.randint(0, 1000)
    #     unique_percentage = 0.2
    #     map_per_threshold = find_threshold(
    #         df=df,
    #         label_column="label_string",
    #         grid_start=7.0,
    #         grid_end=20.0,
    #         grid_num=20,
    #         unique_percentage=unique_percentage,
    #         normalize_label_distribution=True,
    #         seed=None
    #     )
    #     plot_map_vs_thresholds(
    #         map_per_threshold, "Threshold vs mAP", f"mAP_vs_Threshold_random_seed_{i}.png"
    #     )

    #     best_threshold, best_overall_map, new_map, non_new_map, perc_of_new_label = select_best_threshold(map_per_threshold)
    #     print(
    #         f"Best threshold: {best_threshold} with overall mAP: {best_overall_map}, new mAP: {new_map}, non-new mAP: {non_new_map}, perc of new label in query set: {perc_of_new_label}"
    #     )
