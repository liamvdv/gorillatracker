from torchvision.transforms import ToPILImage

from gorillatracker.train_utils import get_data_module


def save_image(tensor, path: str):
    img = ToPILImage()(tensor)
    img.save(path)


def test_offline_data_module():
    dm = get_data_module("gorillatracker.datasets.mnist.MNISTDataset", "./mnist", 1, "offline", lambda x: x)
    dm.setup("fit")
    dl = dm.train_dataloader()
    for i, batch in enumerate(dl):
        images, labels = batch

        a, p, n = images
        # save_image(a[0], f"tests/{i}_anchor.png")
        # save_image(p[0], f"tests/{i}_positive.png")
        # save_image(n[0], f"tests/{i}_negative.png")
        # assert i < 5
        for al, pl, nl in zip(*labels):
            assert al == pl
            assert al != nl
