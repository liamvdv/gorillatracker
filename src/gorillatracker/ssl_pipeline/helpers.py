from __future__ import annotations

from dataclasses import dataclass


def jenkins_hash(key: int) -> int:
    hash_value = ((key >> 16) ^ key) * 0x45D9F3B
    hash_value = ((hash_value >> 16) ^ hash_value) * 0x45D9F3B
    hash_value = (hash_value >> 16) ^ hash_value & 0xFFFFFFFF
    return hash_value


@dataclass(frozen=True)
class BoundingBox:
    x_top_left: int
    y_top_left: int
    x_bottom_right: int
    y_bottom_right: int

    @property
    def top_left(self) -> tuple[int, int]:
        return self.x_top_left, self.y_top_left

    @property
    def bottom_right(self) -> tuple[int, int]:
        return self.x_bottom_right, self.y_bottom_right

    @staticmethod
    def from_yolo(
        x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int
    ) -> BoundingBox:
        x_top_left = int((x_center - width / 2) * image_width)
        y_top_left = int((y_center - height / 2) * image_height)
        x_bottom_right = int((x_center + width / 2) * image_width)
        y_bottom_right = int((y_center + height / 2) * image_height)
        return BoundingBox(x_top_left, y_top_left, x_bottom_right, y_bottom_right)
