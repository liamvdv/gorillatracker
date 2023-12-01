import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torchmetrics as tm
import wandb
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torchmetrics.functional import pairwise_euclidean_distance


def log_as_wandb_table(embeddings_table, run):
    tmp = embeddings_table.apply(
        lambda row: pd.concat([pd.Series([row["label"]]), pd.Series(row["embedding"])]), axis=1
    )
    tmp.columns = ["label"] + [f"embedding_{i}" for i in range(len(embeddings_table["embedding"].iloc[0]))]
    run.log({"embeddings": wandb.Table(dataframe=tmp)})


class LogEmbeddingsToWandbCallback(L.Callback):
    """
    A pytorch lightning callback that saves embeddings to wandb and logs them.

    Args:
        every_n_val_epochs: Save embeddings every n epochs as a wandb artifact (of validation set).
        log_share: Log embeddings to wandb every n epochs.
    """

    def __init__(self, every_n_val_epochs, wandb_run):
        super().__init__()
        self.every_n_val_epochs = every_n_val_epochs
        self.logged_epochs = set()
        self.embedding_artifacts = []
        self.run = wandb_run

    def on_validation_epoch_end(self, trainer, pl_module):
        embeddings_table = pl_module.embeddings_table

        current_epoch = trainer.current_epoch

        if (
            current_epoch % self.every_n_val_epochs == 0
            and current_epoch not in self.logged_epochs
            and current_epoch != 0
        ) or (trainer.max_epochs - 1 == current_epoch):
            self.logged_epochs.add(current_epoch)

            # Assuming you have an 'embeddings' variable containing your embeddings

            table = wandb.Table(columns=embeddings_table.columns.to_list(), data=embeddings_table.values)  # TODO
            artifact = wandb.Artifact(
                name="run_{0}_epoch_{1}".format(self.run.name, current_epoch),
                type="embeddings",
                metadata={"epoch": current_epoch},
                description="Embeddings from epoch {}".format(current_epoch),
            )
            artifact.add(table, "embeddings_table_epoch_{}".format(current_epoch))
            self.run.log_artifact(artifact)
            self.embedding_artifacts.append(artifact.name)
            # log metrics to wandb
            evaluate_embeddings(
                data=embeddings_table,
                embedding_name="val/embeddings",
                metrics={"knn": knn, "pca": pca, "tsne": tsne, "fc_layer": fc_layer},  # "flda": flda_metric,
            )
            # wandb.log({"epoch": current_epoch})
            # for visibility also log the
        # clear the table where the embeddings are stored
        pl_module.embeddings_table = pd.DataFrame(columns=pl_module.embeddings_table_columns)  # reset embeddings table


# now add stuff to evaluate the embeddings / the model that created the embeddings
# 1. add a fully connected layer to the model that takes the embeddings as input and outputs the labels -> then train this model -> evaluate false positive, false negative, accuracy, ...)
# 2. use different kinds of clustering algorithms to cluster the embeddings -> evaluate (false positive, false negative, accuracy, ...)
# 3. use some kind of FLDA ({(m_1 - m_2)^2/(s_1^2 + s_2^2)} like metric to evaluate the quality of the embeddings
# 4. try kNN with different k values to evaluate the quality of the embeddings
# 5. enjoy
def load_embeddings_from_wandb(embedding_name, run):
    """Load embeddings from wandb Artifact."""
    # Data is a pandas Dataframe with columns: label, embedding_0, embedding_1, ... loaded from wandb from the
    artifact = run.use_artifact(embedding_name, type="embeddings")
    data_table = artifact.get("embeddings_table_epoch_10")  # TODO
    data = pd.DataFrame(data=data_table.data, columns=data_table.columns)
    return data


def evaluate_embeddings(data, embedding_name, metrics={}):  # data is DataFrame with columns: label and embedding
    embeddings = np.asarray([embedding for embedding in data["embedding"]], dtype=np.float32)
    # convert labels to numpy array
    le = LabelEncoder()
    labels = le.fit_transform(data["label"])

    # labels = np.asarray([label for label in data["label"]], dtype=np.int32)

    results = {}
    for metric_name, metric in metrics.items():
        # Calculate the metric
        result = metrics[metric_name](embeddings, labels)
        if result.__class__ == dict:
            # log everything to wandb
            for key, value in result.items():
                wandb.log({f"{embedding_name}/{metric_name}/{key}": value})
        else:
            # Log metrics to WandB
            wandb.log({f"{embedding_name}/{metric_name}": result})
        # print(f"{metric_name}: {result}")
        results[metric_name] = result

    return results


def fc_layer(embeddings, labels, batch_size=64, epochs=300, seed=42):
    num_classes = len(np.unique(labels))
    model = torch.nn.Sequential(
        torch.nn.Linear(embeddings.shape[1], 100),
        torch.nn.Sigmoid(),
        torch.nn.Linear(100, num_classes),
    )
    for param in model.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    # acitvate gradients
    with torch.set_grad_enabled(True):
        for epoch in range(epochs):
            loss_sum = 0.0
            embeddings_copy, labels_copy = sklearn.utils.shuffle(
                embeddings, labels, random_state=seed + epoch, n_samples=len(embeddings)
            )

            for i in range(0, len(embeddings_copy), batch_size):
                batch_embeddings = torch.tensor(embeddings_copy[i : i + batch_size])
                batch_labels = torch.tensor(labels_copy[i : i + batch_size], dtype=torch.long)

                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                # apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item()

            loss_mean = loss_sum / (len(embeddings_copy) / batch_size)
            loss_sum = 0.0
            if epoch % 100 == 0:
                print(f"Loss: {loss_mean} in Epoch {epoch}")

    final_outputs = None
    embeddings = torch.tensor(embeddings)
    labels = torch.tensor(labels)
    with torch.no_grad():
        final_outputs = torch.nn.functional.softmax(model(embeddings), dim=1)

    accuracy = tm.functional.accuracy(
        final_outputs, labels, task="multiclass", num_classes=num_classes, average="weighted"
    ).item()
    accuracy_top5 = tm.functional.accuracy(
        final_outputs, labels, task="multiclass", num_classes=num_classes, top_k=5
    ).item()
    auroc = tm.functional.auroc(final_outputs, labels, task="multiclass", num_classes=num_classes).item()
    f1 = tm.functional.f1_score(
        final_outputs, labels, task="multiclass", num_classes=num_classes, average="weighted"
    ).item()
    return {"accuracy": accuracy, "accuracy_top5": accuracy_top5, "auroc": auroc, "f1": f1}


# Vincents code
def knn(embeddings, labels, k=5):
    num_classes = len(np.unique(labels))
    # convert embeddings and labels to tensors
    embeddings = torch.tensor(embeddings)
    labels = torch.tensor(labels)

    distance_matrix = pairwise_euclidean_distance(embeddings)

    # Ensure distances on the diagonal are set to a large value so they are ignored
    distance_matrix.fill_diagonal_(float("inf"))

    # Find the indices of the closest embeddings for each embedding
    classification_matrix = torch.zeros((len(distance_matrix), k))
    for i in range(k):
        closest_indices = torch.argmin(distance_matrix, dim=1)
        closest_labels = labels[closest_indices]
        # Set the distance to the closest embedding to a large value so it is ignored
        distance_matrix[torch.arange(len(distance_matrix)), closest_indices] = float("inf")
        classification_matrix[:, i] = closest_labels
    # Calculate the most common label for each embedding
    # transform classification_matrix of shape (n,k) to (n,num_classes) where num_classes is the number of unique labels
    # the idea is that in the end the classification_matrix contains the probability for each class for each embedding
    classification_matrix_cpy = classification_matrix.clone()
    classification_matrix = torch.zeros((len(classification_matrix), num_classes))
    for i in range(num_classes):
        classification_matrix[:, i] = torch.sum(classification_matrix_cpy == i, dim=1) / k

    accuracy = tm.functional.accuracy(
        classification_matrix, labels, task="multiclass", num_classes=num_classes, average="weighted"
    ).item()
    accuracy_top5 = tm.functional.accuracy(
        classification_matrix, labels, task="multiclass", num_classes=num_classes, top_k=5
    ).item()
    auroc = tm.functional.auroc(classification_matrix, labels, task="multiclass", num_classes=num_classes).item()
    f1 = tm.functional.f1_score(
        classification_matrix, labels, task="multiclass", num_classes=num_classes, average="weighted"
    ).item()
    print("knn done")
    return {"accuracy": accuracy, "accuracy_top5": accuracy_top5, "auroc": auroc, "f1": f1}


def pca(embeddings, labels):  # generate a 2D plot of the embeddings
    num_classes = len(np.unique(labels))
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)
    # plot embeddings

    plt.figure()
    plot = sns.scatterplot(
        x=embeddings[:, 0], y=embeddings[:, 1], palette=sns.color_palette("hls", num_classes), hue=labels
    )
    # ignore outliers when calculating the axes limits
    x_min, x_max = np.percentile(embeddings[:, 0], [0.1, 99.9])
    y_min, y_max = np.percentile(embeddings[:, 1], [0.1, 99.9])
    plot.set_xlim(x_min, x_max)
    plot.set_ylim(y_min, y_max)
    plot.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
    # plot.figure.savefig("pca.png")
    plot = wandb.Image(plot.figure)
    print("pca done")
    return plot


def tsne(embeddings, labels, pca=False, count=1000):  # generate a 2D plot of the embeddings
    num_classes = len(np.unique(labels))
    # downsample the embeddings and also the labels to 1000 samples
    indices = np.random.choice(len(embeddings), min(count, len(labels)), replace=False)
    embeddings = embeddings[indices]
    labels = labels[indices]
    if len(labels) < 50:
        return
    if pca:
        pca = sklearn.decomposition.PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)

    # tsne = TSNE(n_components=2, method="exact")
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure()
    plot = sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        palette=sns.color_palette("hls", num_classes),
        hue=labels,
    )
    # ignore outliers when calculating the axes limits
    x_min, x_max = np.percentile(embeddings[:, 0], [1, 99])
    y_min, y_max = np.percentile(embeddings[:, 1], [1, 99])
    plot.set_xlim(x_min, x_max)
    plot.set_ylim(y_min, y_max)
    # place the legend outside of the plot but readable
    plot.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
    # plot.figure.savefig("tnse.png")
    plot = wandb.Image(plot.figure)
    print("tsne done")
    return plot


def flda_metric(embeddings, labels):  # TODO: test
    # num_classes = len(np.unique(labels))
    # (m_1 - m_2)^2/(s_1^2 + s_2^2)
    mean_var_map = {label: [0.0, 0.0] for label in np.unique(labels)}
    ratio_sum = 0.0

    for label in np.unique(labels):
        class_embeddings = embeddings[labels == label]
        mean = np.mean(class_embeddings, axis=0)
        variance = np.var(class_embeddings, axis=0)
        mean_var_map[label] = [mean, variance]

    for label in np.unique(labels):
        for label2 in np.unique(labels):
            if label == label2:
                continue
            mean1, var1 = mean_var_map[label]
            mean2, var2 = mean_var_map[label2]
            # mean and variance are vectors so use euclidean distance
            ratio = np.linalg.norm(mean1 - mean2) / (np.linalg.norm(var1) + np.linalg.norm(var2))
            # ratio = np.(mean1 - mean2) / (var1 + var2)
            ratio_sum += ratio

    return ratio_sum / 2  # because we calculate the ratio twice for each pair


def kmeans(embeddings, num_clusters=2):  # TODO: log some kind of normalized mutual information score
    k_means = sklearn.cluster.KMeans(n_clusters=num_clusters)
    outputs = k_means.fit_predict(embeddings)
    return k_means.cluster_centers_, outputs


if __name__ == "__main__":
    # Test the EmbeddingAnalyzer and Accuracy metric
    run = wandb.init(entity="gorillas", project="MNIST-EfficientNetV2", name="test_embeddings2")
    data = load_embeddings_from_wandb("run_MNISTTest5-2023-11-11-15-17-17_epoch_10:v0", run)
    results = evaluate_embeddings(
        data=data,
        embedding_name="run_MNISTTest5-2023-11-11-15-17-17_epoch_10:v0",
        metrics={
            "pca": pca,
            "tsne": tsne,
            "knn": knn,
            "flda": flda_metric,
            "fc_layer": fc_layer,
            # "kmeans": kmeans
        },
    )
    print(results)
