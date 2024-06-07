from PIL.Image import Image
from torchvision.transforms.functional import pad


class SquarePad:
    def __call__(self, image: Image) -> Image:
        # calc padding
        width, height = image.size
        aspect_ratio = width / height
        if aspect_ratio > 1:
            padding_top = (width - height) // 2
            padding_bottom = width - height - padding_top
            padding = (0, padding_top, 0, padding_bottom)
        else:
            padding_left = (height - width) // 2
            padding_right = height - width - padding_left
            padding = (padding_left, 0, padding_right, 0)

        image = pad(image, padding, (0, 0, 0), "constant")
        return image
