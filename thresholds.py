from gorillatracker.utils.embedding_generator import generate_embeddings_from_run, read_embeddings_from_disk
import numpy as np
from folds import DistinctClassFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, pairwise_distances
import tqdm
import matplotlib.pyplot as plt


def find_threshold(
    df, label_column: str, grid_start: float, grid_end: float, grid_num: float, unique_percentage=0.2, seed=42
):
    # Thresholds for grid search
    thresholds = np.linspace(grid_start, grid_end, grid_num)

    # For storing results
    results = []
    kf = DistinctClassFold(n_buckets=5, shuffle=True, random_state=seed)

    # TODO(liamvdv): handle no sufficient match (singletons cannot have match)
    new_label = "new"
    for dataset_df, query_df in kf.split(
        df, label_column="label_string", label_mask=new_label, unique_percentage=unique_percentage
    ):
        # Compute pairwise distances between query and training embeddings
        query = np.stack(query_df["embedding"].values)
        dataset = np.stack(dataset_df["embedding"].values)
        distances = pairwise_distances(query, dataset)
        closest_indices = np.argmin(distances, axis=1)

        # Now you can get the predicted label based on the closest train embedding
        predicted_labels = dataset_df.iloc[closest_indices]["label_string"].values
        actual_labels = query_df["label_string"].values

        # Binarize the labels for mAP calculation
        combined_labels = np.concatenate([df["label_string"], [new_label]])
        unique_labels = np.unique(combined_labels)
        # NOTE(liamvdv): creates a len(actual_labels) x len(unique_labels) matrix with 0 or 1 values
        Y_true_bin = label_binarize(actual_labels, classes=unique_labels)

        for threshold in tqdm.tqdm(thresholds, "Grid Search Thresholds"):
            thresholded_labels = [
                "new" if distances[i][closest_indices[i]] > threshold else predicted_label
                for i, predicted_label in enumerate(predicted_labels)
            ]
            # Binarize the labels for mAP calculation
            Y_pred_bin = label_binarize(thresholded_labels, classes=unique_labels)

            # Calculate mAP for all labels
            map_score_all = average_precision_score(Y_true_bin, Y_pred_bin, average="macro")

            # Calculate mAP for the 'new' label
            assert "new" in unique_labels
            new_index = list(unique_labels).index("new")
            # [:,<idx>] only selects a given column, here the "new" column
            map_score_new = average_precision_score(Y_true_bin[:, new_index], Y_pred_bin[:, new_index])

            # Calculate mAP for non-new labels
            non_new_indices = [i for i, label in enumerate(unique_labels) if label != new_label]
            map_score_non_new = average_precision_score(
                Y_true_bin[:, non_new_indices], Y_pred_bin[:, non_new_indices], average="macro"
            )

            results.append((threshold, map_score_all, map_score_new, map_score_non_new))

        results = np.array(results)
        unique_thresholds = np.unique(results[:, 0])
        map_per_threshold = {
            threshold: np.mean(results[results[:, 0] == threshold], axis=0)[1:] for threshold in unique_thresholds
        }
        return map_per_threshold


def plot_map_vs_thresholds(map_per_threshold: dict, title: str, path: str):
    # Extracting the data for plotting
    thresholds = list(map_per_threshold.keys())
    mAP_overall = [values[0] for values in map_per_threshold.values()]
    mAP_new = [values[1] for values in map_per_threshold.values()]
    mAP_non_new = [values[2] for values in map_per_threshold.values()]

    # Creating the plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mAP_overall, label="Overall", marker="o")
    plt.plot(thresholds, mAP_new, label="Only New", marker="o")
    plt.plot(thresholds, mAP_non_new, label="Only Non-New", marker="o")

    # Adding title and labels
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("mAP")
    plt.legend()

    # Showing the plot
    plt.grid(True)
    plt.savefig(path)


if __name__ == "__main__":
    # generate_embeddings_from_run("https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/cc6tiy3f/workspace", "example-embeddings.pkl")
    df = read_embeddings_from_disk("example-embeddings.pkl")
    # print(df.head())
    # Convert embeddings to numpy arrays if they are not already
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))

    unique_percentage = 0.2
    map_per_threshold = find_threshold(
        df=df,
        label_column="label_string",
        grid_start=7.0,
        grid_end=20.0,
        grid_num=20,
        unique_percentage=unique_percentage,
    )
    plot_map_vs_thresholds(
        map_per_threshold, f"Threshold vs mAP @ {unique_percentage * 100}% new individuals", "mAP_vs_Threshold.png"
    )

    best_threshold, best_overall_map, new_map, non_new_map = max(
        [[label, *mAPs] for label, mAPs in map_per_threshold.items()], key=lambda x: x[1]
    )
    print(
        f"Best threshold: {best_threshold} with overall mAP: {best_overall_map}, new mAP: {new_map}, non-new mAP: {non_new_map}"
    )
