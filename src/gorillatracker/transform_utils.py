import torch
from torchvision.transforms.functional import pad


class SquarePad:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return pad(image, padding, 0, "constant")
