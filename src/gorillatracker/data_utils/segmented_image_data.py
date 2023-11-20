import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SegmentedImageData:
    path: str
    segments: Dict[str, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = field(default_factory=dict)

    def add_segment(self, class_label: str, mask: np.ndarray, box: Tuple[int, int, int, int]):
        """
        class_label: label of the segment
        mask: binary mask of the segment
        box: bounding box of the segment in x_min, y_min, x_max, y_max format
        """
        if class_label not in self.segments:
            self.segments[class_label] = []
        self.segments[class_label].append((mask, box))

    @property
    def filename(self) -> str:
        return os.path.basename(self.path).split(".")[0]
