from collections import defaultdict
from typing import Protocol

from gorillatracker.ssl_pipeline.helpers import AssociatedBoundingBox, BoundingBox


class Correlator(Protocol):
    def __call__(
        self,
        reference_boxes: list[AssociatedBoundingBox],
        unassociated_boxes: list[BoundingBox],
        threshold: float = 0.7,
    ) -> tuple[list[AssociatedBoundingBox], list[BoundingBox]]:
        """
        Protocol for correlators.

        Args:
            reference_boxes:
                List of bounding boxes used as reference for the correlation.
            unassociated_boxes:
                List of bounding boxes to be correlated with the `reference_boxes`.
            threshold:
                Threshold for the intersection over union (IoU) metric.

        Returns:
            A tuple containing two lists:
            - The first list contains the associated bounding boxes. These are the
              bounding boxes from `unassociated_boxes` that have been successfully
              associated with bounding boxes from `associated_boxes`.
            - The second list contains the remaining unassociated bounding boxes. These
              are the bounding boxes from `unassociated_boxes` for which a relationship
              could not be resolved with any of the bounding boxes in `reference_boxes`.
        """
        ...


ReferenceToUnassociated = dict[AssociatedBoundingBox, list[BoundingBox]]
UnassociatedToReference = dict[BoundingBox, list[AssociatedBoundingBox]]


def build_intersection_graph(
    reference_boxes: list[AssociatedBoundingBox], unassociated_boxes: list[BoundingBox], threshold: float = 0.7
) -> tuple[ReferenceToUnassociated, UnassociatedToReference]:
    """Build the intersection graph between associated and unassociated boxes."""

    rtu: ReferenceToUnassociated = defaultdict(list)
    utr: UnassociatedToReference = defaultdict(list)

    for r_box in reference_boxes:
        for u_box in unassociated_boxes:
            if r_box.bbox.iou(u_box) > threshold:
                rtu[r_box].append(u_box)
                utr[u_box].append(r_box)
    return rtu, utr


def one_to_one_resolver(
    reference_to_unassociated: ReferenceToUnassociated, unassociated_to_reference: UnassociatedToReference
) -> list[AssociatedBoundingBox]:
    """Resolve the one-to-one association between reference and unassociated boxes."""
    return [
        AssociatedBoundingBox(r_boxes[0].association, u_box)
        for u_box, r_boxes in unassociated_to_reference.items()
        if len(r_boxes) == 1 and len(reference_to_unassociated[r_boxes[0]]) == 1  # one to one relationship
    ]


def one_to_one_correlator(
    reference_boxes: list[AssociatedBoundingBox], unassociated_boxes: list[BoundingBox], threshold: float = 0.7
) -> tuple[list[AssociatedBoundingBox], list[BoundingBox]]:
    """Correlate associated and unassociated boxes which have an one-to-one relationship."""
    assert 0 <= threshold <= 1, "Threshold must be between 0 and 1"
    reference_to_unassociated, unassociated_to_reference = build_intersection_graph(
        reference_boxes, unassociated_boxes, threshold
    )
    associated_boxes = one_to_one_resolver(reference_to_unassociated, unassociated_to_reference)
    unresolved_boxes = [box for box in unassociated_boxes if box not in [a_box.bbox for a_box in associated_boxes]]
    return associated_boxes, unresolved_boxes
