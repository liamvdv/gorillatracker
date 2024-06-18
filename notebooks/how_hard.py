from gorillatracker.ssl_pipeline.ssl_config import SSLConfig
from pathlib import Path


ssl_config = SSLConfig(
    tff_selection="equidistant",
    negative_mining="random",
    n_samples=10000,
    feature_types=["face_90"],
    min_confidence=0.6,
    min_images_per_tracking=100,
    width_range=(40, None),
    height_range=(40, None),
    split_path=Path(
        "/workspaces/gorillatracker/data/splits/SSL/SSL-Video-Split_2024-04-18_percentage-80-10-10_split.pkl"
    ),
)

import time

before = time.time()
contrastive_sampler = ssl_config.get_contrastive_sampler(Path("cropped-images/2024-04-18"), "train")
after = time.time()
print(f"Time: {after - before}")
print(len(contrastive_sampler))
for i in range(10):
    contrastive_image = contrastive_sampler[i * 10]
    print(contrastive_image)
    print(contrastive_sampler.positive(contrastive_image))
    print(contrastive_sampler.negative(contrastive_image))


positives_with_dist = contrastive_sampler.positives_with_dist(contrastive_image)
