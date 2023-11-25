import pytorch_lightning as pl
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam
from torch.utils.data import DataLoader

from gorillatracker.cvat_import import cvat_import
from gorillatracker.datasets.sam_cxl import SAMCXLDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, cvat_path: str, img_path: str, batch_size: int, sam_model: Sam):
        super().__init__()
        self.cvat_path = cvat_path
        self.img_path = img_path
        self.batch_size = batch_size
        self.sam_model = sam_model

    def setup(self, stage: str):
        segmented_gorilla_images = cvat_import(self.cvat_path, self.img_path)
        if stage == "fit":
            self.train = SAMCXLDataset(segmented_gorilla_images, "train", self.sam_model)
            self.val = SAMCXLDataset(segmented_gorilla_images, "val", self.sam_model)
        elif stage == "test":
            self.test = SAMCXLDataset(segmented_gorilla_images, "test", self.sam_model)
        elif stage == "validate":
            self.val = SAMCXLDataset(segmented_gorilla_images, "val", self.sam_model)
        elif stage == "predict":
            # TODO(memben): implement this
            self.predict = None
            raise ValueError("stage predict not yet supported by data module.")
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


if __name__ == "__main__":
    base_path = "/workspaces/gorillatracker/data/ground_truth/cxl"
    cvat_path = f"{base_path}/full_images_body_instance_segmentation/cvat_export.xml"
    img_path = f"{base_path}/full_images/"
    model_type = "vit_h"
    checkpoint_path = "/workspaces/gorillatracker/models/sam_vit_h_4b8939.pth"
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    batch_size = 32
    dm = SegmentationDataModule(cvat_path, img_path, batch_size, sam_model)
    dm.setup("test")
    for batch in dm.test_dataloader():
        print(batch)
        break
