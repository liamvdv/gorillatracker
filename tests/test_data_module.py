import torch
from torchvision.transforms import ToPILImage

from gorillatracker.train_utils import get_data_module


def save_image(tensor: torch.Tensor, path: str) -> None:
    img = ToPILImage()(tensor)
    img.save(path)


def test_offline_data_module() -> None:
    dm = get_data_module(
        "gorillatracker.datasets.mnist.MNISTDataset",
        "./mnist",
        1,
        "offline",
        lambda x: x,
        training_transforms=lambda x: x,
    )
    dm.setup("fit")
    dl = dm.train_dataloader()
    for i, batch in enumerate(dl):
        ids, images, labels = batch

        a, p, n = images
        # save_image(a[0], f"tests/{i}_anchor.png")
        # save_image(p[0], f"tests/{i}_positive.png")
        # save_image(n[0], f"tests/{i}_negative.png")
        # assert i < 5
        for al, pl, nl in zip(*labels):
            assert al == pl
            assert al != nl
