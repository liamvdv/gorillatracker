import torch
from torchvision import transforms

from gorillatracker.transform_utils import SquarePad


def get_model_input(
    dataset_cls, dataset_path: str, partion: str = "train", amount_of_tensors: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a tensor of images and a tensor of labels from a dataset.
    Args:
        dataset_cls: The dataset class to use.
        dataset_path: The path to the dataset.
        partion: The partion of the dataset to use.
        amount_of_tensors: The amount of tensors to get from the dataset. If -1, get all tensors.
    """

    dataset = dataset_cls(
        dataset_path,
        partion,
        transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    if amount_of_tensors == -1:
        amount_of_tensors = len(dataset)

    images = []
    labels = []
    for i in range(amount_of_tensors):
        _, image, label = dataset[i]
        images.append(image)
        labels.append(label)

    return torch.stack(images), torch.tensor(labels)


def log_model_to_file(model, file_name) -> None:
    with open("model.txt", "w") as f:
        f.write(str(model))


def print_model_parameters(model) -> None:
    for name, param in model.named_parameters():
        print(f"{name}: {param.nelement()} parameters")
