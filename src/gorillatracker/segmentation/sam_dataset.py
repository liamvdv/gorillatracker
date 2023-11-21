from typing import List

import cv2
import pytorch_lightning as pl
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset

from gorillatracker.cvat_import import SegmentedImageData

# baseline https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/


class SegmentationDataset(Dataset):
    def __init__(self, segmented_images: List[SegmentedImageData], sam_model, device):
        self.data = self._preprocess_data(segmented_images, sam_model, device)

    def _preprocess_data(self, segmented_images, sam_model, device):
        data = []
        transform = ResizeLongestSide(sam_model.image_encoder.img_size)

        for img in segmented_images:
            input_image, input_size, original_image_size = self._preprocess_image(
                img.path, sam_model, device, transform
            )

            image_embedding = self._embed_image(input_image, sam_model)
            for box, mask in zip(img.boxes, img.masks):
                sparse_embeddings, dense_embeddings = self._embed_prompt(
                    box, original_image_size, sam_model, device, transform
                )

                data.append(
                    [
                        image_embedding,
                        input_size,
                        original_image_size,
                        sparse_embeddings,
                        dense_embeddings,
                        mask,
                    ]
                )

    def _preprocess_image(self, img_path, sam_model, device, transform):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])
        input_image = input_image.to(device)
        return input_image, input_size, original_image_size

    def _embed_image(self, image, sam_model):
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(image)
            return image_embedding

    def _embed_prompt(self, prompt_box, original_image_size, sam_model, device, transform):
        with torch.no_grad():
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
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


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
