from typing import List, Literal

import cv2
import numpy as np
import torch
from segment_anything.modeling.sam import Sam
from segment_anything.utils.transforms import ResizeLongestSide
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# test
from gorillatracker.cvat_import import SegmentedImageData

# baseline https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/


class SAMCXLDataset(Dataset):
    def __init__(
        self, segmented_images: List[SegmentedImageData], partition: Literal["train", "val", "test"], sam_model: Sam
    ):
        """
        Dataset of the segmented cxl images, with embeddings precomputed with SAM on GPU.
        The dataset should be used to fine-tune the SAM decoder.
        Deterministic but random split of segmented_images into 70% train, 15% val, 15% test.

        Args:
        segmented_images (List[SegmentedImageData]): A list of segmented image data.
        partition (Literal["train", "val", "test"]): The dataset partition to be used; one of "train", "val", or "test".
        sam_model (Sam): An instance of the SAM model used for processing the images.

        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sam_model.to(self.device)

        train, temp = train_test_split(segmented_images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        partition_mapping = {"train": train, "val": val, "test": test}
        if partition not in partition_mapping:
            raise ValueError(f"partition must be one of ['train', 'val', 'test'], got {partition}")

        self.data = self._preprocess_data(partition_mapping[partition], sam_model)

    def _preprocess_data(self, segmented_images: List[SegmentedImageData], sam_model: Sam):
        transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        return [
            data_item for img in segmented_images for data_item in self._process_single_image(img, sam_model, transform)
        ]

    def _process_single_image(self, img: SegmentedImageData, sam_model: Sam, transform):
        input_image, input_size, original_image_size = self._preprocess_image(img.path, sam_model, transform)
        image_embedding = self._embed_image(input_image, sam_model)

        return [
            [
                image_embedding,
                input_size,
                original_image_size,
                *self._embed_prompt(np.array(box), original_image_size, sam_model, transform),
                mask,
            ]
            for class_label, segment_list in img.segments.items()
            for mask, box in segment_list
        ]

    def _preprocess_image(self, img_path: str, sam_model: Sam, transform):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image.astype("float32")).to(self.device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])
        return input_image, input_size, original_image_size

    def _embed_image(self, image, sam_model: Sam):
        # image = image.to(self.device)
        with torch.no_grad():
            return sam_model.image_encoder(image)

    def _embed_prompt(self, prompt_box, original_image_size, sam_model: Sam, transform):
        with torch.no_grad():
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float).to(self.device)
            box_torch = box_torch[None, :]

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            return sparse_embeddings, dense_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
