from typing import List

import numpy as np
import cv2
import pytorch_lightning as pl
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset

from gorillatracker.cvat_import import SegmentedImageData, cvat_import

# baseline https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/


class SegmentationDataset(Dataset):
    def __init__(self, segmented_images: List[SegmentedImageData], sam_model):
        self.data = self._preprocess_data(segmented_images, sam_model)

    def _preprocess_data(self, segmented_images, sam_model):
        data = []
        transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        
        for img in segmented_images:
            input_image, input_size, original_image_size = self._preprocess_image(img.path, sam_model, transform)

            image_embedding = self._embed_image(input_image, sam_model)
            for class_label, segment_list in img.segments.items():
                for mask, box in segment_list:
                    box = np.array(box)
                    sparse_embeddings, dense_embeddings = self._embed_prompt(box, original_image_size, sam_model, transform)

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
        return data

    def _preprocess_image(self, img_path, sam_model, transform):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image.astype("float32"))
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])
        return input_image, input_size, original_image_size

    def _embed_image(self, image, sam_model):
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(image)
            return image_embedding

    def _embed_prompt(self, prompt_box, original_image_size, sam_model, transform):
        with torch.no_grad():
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float)
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
    def __init__(self):
        super().__init__()
        base_path = "/workspaces/gorillatracker/data/ground_truth/cxl"
        cvat_path = f"{base_path}/full_images_body_instance_segmentation/cvat_export.xml"
        img_path = f"{base_path}/full_images/"
        segmented_gorilla_images = cvat_import(cvat_path, img_path)
        np.random.seed(50)
        np.random.shuffle(segmented_gorilla_images)
        segmented_gorilla_images = segmented_gorilla_images[:1]
        model_type = "vit_h"
        checkpoint_path = "/workspaces/gorillatracker/models/sam_vit_h_4b8939.pth"
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.dataset = SegmentationDataset(segmented_gorilla_images, sam_model)
        self.batch_size = 4

    @classmethod
    def from_training_args(cls, args):
        return cls()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self): 
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    

if __name__ == "__main__":
    dm = SegmentationDataModule()
    test = iter(dm.train_dataloader())
    # _check_dataloader_iterable
    
    
    for batch in dm.train_dataloader():
        print(batch)
        break