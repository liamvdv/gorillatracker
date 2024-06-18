from sqlalchemy import create_engine
from gorillatracker.data.nlet import NletDataModule, build_onelet
from gorillatracker.data.utils import flatten_batch
from gorillatracker.data.ssl import SSLDataset
from gorillatracker.ssl_pipeline.dataset import GorillaDatasetKISZ
from gorillatracker.ssl_pipeline.ssl_config import SSLConfig
from sqlalchemy.orm import sessionmaker
from gorillatracker.ssl_pipeline.models import Video
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

from scipy.spatial import distance
from gorillatracker.ssl_pipeline.dataset_splitter import SplitArgs
import numpy as np
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2

from pathlib import Path
import time


def embedding_generation():
    split_path = Path(
        "/workspaces/gorillatracker/data/splits/SSL/SSL-Video-Split_2024-04-18_percentage-80-10-10_split.pkl"
    )

    config = SSLConfig(
        tff_selection="random",
        negative_mining="random",
        n_samples=4000,
        feature_types=["body", "face_45", "face_90"],
        min_confidence=0.0,
        min_images_per_tracking=0,
        split_path=split_path,
        width_range=(None, None),
        height_range=(None, None),
    )

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    transform = transforms_v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms_v2.Resize((224, 224))
    transform = transforms_v2.Compose([resize, transform])

    datamodule = NletDataModule(
        Path("/workspaces/gorillatracker/cropped-images/2024-04-18"),
        SSLDataset,
        build_onelet,
        512,
        12,
        model_transforms=transform,
        training_transforms=lambda x: x,
        dataset_names=["SSLDataset"],
        ssl_config=config,
    )

    start_time = time.time()
    datamodule.setup("fit")
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    numpy_embeddings = []
    numpy_ids = []

    for batch in datamodule.train_dataloader():
        print("batch")
        ids, images, labels = flatten_batch(batch)
        db_ids = [int(Path(id).stem) for id in ids]
        embeddings = model(images)
        numpy_embeddings.append(embeddings.detach().numpy())
        numpy_ids.append(db_ids)

    # Convert lists to numpy arrays
    numpy_embeddings = np.concatenate(numpy_embeddings, axis=0)

    numpy_ids = np.concatenate(numpy_ids, axis=0)

    # Save numpy arrays to files
    np.save("numpy_embeddings.npy", numpy_embeddings)
    np.save("numpy_ids.npy", numpy_ids)

    print("Embeddings and IDs have been saved to files.")


def max_min_dist_ranking(embeddings: np.ndarray):
    if embeddings.shape[0] == 0:
        return np.array([])
    if embeddings.shape[0] == 1:
        return np.array([0])

    #  metric k-center
    n_samples = embeddings.shape[0]
    # Calculate the pairwise distance matrix
    dist_matrix = distance.cdist(embeddings, embeddings, "euclidean")

    # Initialize the selected points with the first point
    selected_points = [0]
    remaining_points = set(range(1, n_samples))

    while remaining_points:
        # Find the point that maximizes the minimum distance to the selected points
        max_min_dist = -np.inf
        best_point = None

        for point in remaining_points:
            min_dist = min(dist_matrix[point][i] for i in selected_points)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_point = point

        selected_points.append(best_point)
        remaining_points.remove(best_point)

    return np.array(selected_points)


def max_min_dist_calc():
    split_path = "/workspaces/gorillatracker/data/splits/SSL/SSL-Video-Split_2024-04-18_percentage-80-10-10_split.pkl"

    video_ids = SplitArgs.load_pickle(split_path).train_video_ids()
    engine = create_engine(GorillaDatasetKISZ.DB_URI)

    Session = sessionmaker(bind=engine)
    embeddings = np.load("numpy_embeddings.npy")
    ids = np.load("numpy_ids.npy")

    with Session() as session:
        for video_id in video_ids:
            video = session.get(Video, video_id)
            assert video is not None, f"Video with ID {video_id} not found in the database."
            trackings = video.trackings
            for tracking in trackings:
                tffs = tracking.frame_features
                for feature in ["body", "face_45", "face_90"]:
                    tffs_ids = [tff.tracking_frame_feature_id for tff in tffs if tff.feature_type == feature]
                    tffs_embeddings = embeddings[np.isin(ids, tffs_ids)]
                    if tffs_embeddings.shape[0] == 0:
                        continue
                    tffs_embeddings = tffs_embeddings.reshape(tffs_embeddings.shape[0], -1)  # Reshape to 2D
                    print(tffs_embeddings.shape)
                    ranking = max_min_dist_ranking(tffs_embeddings)
                    print(ranking)


if __name__ == "__main__":
    max_min_dist_calc()
