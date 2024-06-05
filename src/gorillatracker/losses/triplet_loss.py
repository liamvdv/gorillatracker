import logging
from typing import Any, Callable, Dict, List, Literal, Union

import torch
import torch.nn.functional as F
from torch import nn

import gorillatracker.type_helper as gtypes
import gorillatracker.utils.l2sp_regularisation as l2
from gorillatracker.losses.arcface_loss import ArcFaceLoss, VariationalPrototypeLearning
from gorillatracker.losses.offline_distillation_loss import OfflineResponseBasedLoss

eps = 1e-16  # an arbitrary small value to be used for numerical stability tricks

logger = logging.getLogger(__name__)


def get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Compute a mask for valid triplets

    Args:
        labels: Batch of integer labels. shape: (batch_size,)

    Returns:
        Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
        A triplet is valid if:
        `labels[i] == labels[j] and labels[i] != labels[k]`
        and `i`, `j`, `k` are different.
    """
    # step 1 - get a mask for distinct indices

    # shape: (batch_size, batch_size)

    batch_size = labels.size()[0]
    indices_equal = torch.eye(batch_size, dtype=torch.bool)
    indices_not_equal = torch.logical_not(indices_equal)
    # shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2).repeat(1, 1, batch_size)
    # shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1).repeat(1, batch_size, 1)
    # shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0).repeat(batch_size, 1, 1)
    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(
        torch.logical_and(i_not_equal_j, i_not_equal_k),
        j_not_equal_k,
    )

    # step 2 - get a mask for valid anchor-positive-negative triplets

    # shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_equal = labels_equal.to(distinct_indices.device)
    # shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2).repeat(1, 1, batch_size)
    # shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1).repeat(1, batch_size, 1)
    # shape: (batch_size, batch_size, batch_size)

    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # step 3 - combine two masks

    mask = torch.logical_and(distinct_indices, valid_indices)

    return mask


def get_distance_mask(labels: torch.Tensor, valid: Literal["pos", "neg"] = "neg") -> torch.Tensor:
    """Compute mask for the calculation of the hardest positive and negative distance

    Args:
        labels: Batch of labels. shape: (batch_size,)
        valid: whether to calculate the mask for positive or negative distances

    Returns:
        Mask tensor to indicate which distances are actually valid negative or positive distances. Shape: (batch_size, batch_size)
        A positive distance is valid if:
        `labels[i] == labels[j] and i != j`
        A negative distance is valid if:
        `labels[i] != labels[j] and i != j`
    """
    tensor_labels = labels.detach().clone()
    batch_size = tensor_labels.size()[0]
    indices_equal = torch.eye(batch_size, dtype=torch.bool, device=tensor_labels.device)
    indices_not_equal = torch.logical_not(indices_equal)

    if valid == "pos":
        labels_equal = tensor_labels.unsqueeze(0) == tensor_labels.unsqueeze(1)
        mask = torch.logical_and(labels_equal, indices_not_equal)
    elif valid == "neg":
        labels_not_equal = tensor_labels.unsqueeze(0) != tensor_labels.unsqueeze(1)
        mask = torch.logical_and(labels_not_equal, indices_not_equal)

    return mask


def get_semi_hard_mask(
    labels: torch.Tensor,
    distance_matrix: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Compute mask for the calculation of the semi-hard triplet loss

    Args:
        labels: Batch of labels. shape: (batch_size,)
        distance_matrix: Batch of distances. shape: (batch_size, batch_size)

    Returns:
        Mask tensor to indicate which distances are actually valid semi-hard distances. Shape: (batch_size, batch_size, batch_size)
        A distance is semi-hard if:
        `labels[i] == labels[j] and labels[i] != labels[k] and distance_matrix[i][j] < distance_matrix[i][k]`
    """
    # filter out all where the distance to a negative is smaller than the max distance to a positive
    device = distance_matrix.device
    tensor_labels = labels.detach().clone()
    tensor_labels = tensor_labels.to(device)
    batch_size = tensor_labels.size()[0]
    indices_equal = torch.eye(batch_size, dtype=torch.bool, device=device)
    indices_not_equal = torch.logical_not(
        indices_equal,
    )
    labels_equal = (tensor_labels.unsqueeze(0) == tensor_labels.unsqueeze(1)).to(device)
    labels_not_equal = torch.logical_not(labels_equal)

    distance_matrix_pos = distance_matrix * torch.logical_and(labels_equal, indices_not_equal).float()
    distance_matrix_neg = distance_matrix * torch.logical_and(labels_not_equal, indices_not_equal).float()

    # filter out all points where the distance to a negative is smaller than the max distance to a positive
    distance_difference = distance_matrix_neg.unsqueeze(1).repeat(1, batch_size, 1) - distance_matrix_pos.unsqueeze(
        2
    ).repeat(
        1, 1, batch_size
    )  # shape: (anchor: batch_size,positive: batch_size, negative: batch_size)

    # filter out all points where the distance to a negative is smaller than distance to a positive
    # now only the triplets where dist_pos < dist_neg are left
    mask = get_triplet_mask(labels)
    semi_hard_mask = distance_difference > 0.0
    semi_hard_mask = semi_hard_mask.to(mask.device)

    return torch.logical_and(mask, semi_hard_mask)


def euclidean_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Efficient computation of Euclidean distance matrix

    Args:
    x: Input tensor of shape (batch_size, embedding_dim)

    Returns:
    Distance matrix of shape (batch_size, batch_size)
    """
    # step 1 - compute the dot product

    # shape: (batch_size, batch_size)
    dot_product = torch.mm(embeddings, embeddings.t())

    # step 2 - extract the squared Euclidean norm from the diagonal

    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)

    # step 3 - compute squared Euclidean distances

    # shape: (batch_size, batch_size)
    distance_matrix = F.relu(squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1))

    # step 4 - compute the non-squared distances

    # handle numerical stability
    # derivative of the square root operation applied to 0 is infinite
    # we need to handle by setting any 0 to eps
    mask = (distance_matrix == 0.0).float()  # performed on tensors, __eq__ overwritten

    # use this mask to set indices with a value of 0 to eps
    distance_matrix_stable = torch.sqrt(distance_matrix + mask * eps) * (1.0 - mask)
    return distance_matrix_stable


class TripletLossOnline(nn.Module):
    """
    TripletLossOnline operates on Quadlets and does batch optimization.
    Inspiration:
        https://arxiv.org/pdf/1503.03832.pdf
        https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905

    Args:
      margin: Margin value in the Triplet Loss equation
    """

    def __init__(self, margin: float = 1.0, mode: Literal["hard", "semi-hard", "soft"] = "semi-hard") -> None:
        super().__init__()
        self.margin = margin
        self.mode = mode

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, images: torch.Tensor = torch.Tensor()
    ) -> gtypes.LossPosNegDist:
        """computes loss value.

        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)

        Returns:
          Scalar loss value.
        """

        # step 1 - get distance matrix
        # shape: (batch_size, batch_size)
        distance_matrix = euclidean_distance_matrix(embeddings)

        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

        # shape: (batch_size, batch_size, 1)
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        # shape: (batch_size, 1, batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

        # step 3 - filter out invalid or easy triplets by setting their loss values to 0

        # shape: (batch_size, batch_size, batch_size)
        # we only want to keep correct and depending on the mode the hardest or semi-hard triplets
        # therefore we create a mask that is 1 for all valid triplets and 0 for all invalid triplets
        mask = self.get_mask(distance_matrix, anchor_positive_dists, anchor_negative_dists, labels)
        mask = mask.to(triplet_loss.device)  # ensure mask is on the same device as triplet_loss
        triplet_loss *= mask

        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging
        num_losses = torch.sum(mask)
        triplet_loss = triplet_loss.sum() / (num_losses + eps)

        # calculate the average positive and negative distance
        anchor_positive_dist_sum = (anchor_positive_dists.repeat(1, 1, len(labels)) * mask).sum()
        anchor_negative_dist_sum = (anchor_negative_dists.repeat(1, len(labels), 1) * mask).sum()
        anchor_positive_dist_mean = anchor_positive_dist_sum / (num_losses + eps)
        anchor_negative_dist_mean = anchor_negative_dist_sum / (num_losses + eps)

        return triplet_loss, anchor_positive_dist_mean, anchor_negative_dist_mean

    def get_mask(
        self,
        distance_matrix: torch.Tensor,
        anchor_positive_dists: torch.Tensor,
        anchor_negative_dists: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        mask = get_triplet_mask(labels)
        if self.mode == "hard":  # take only the hardest negative as a negative per anchor
            neg_mask = get_distance_mask(labels, valid="neg")  # get all valid negatives

            # for each anchor compute the min distance to a negative
            masked_anchor_negative_dists = anchor_negative_dists.squeeze(1).masked_fill(
                neg_mask == 0, float("inf")
            )  # fill all invalid negatives with inf so they are not considered in the min
            _, neg_min_indices = torch.min(masked_anchor_negative_dists, dim=1)
            # print(neg_min_indices)

            pos_mask = get_distance_mask(labels, valid="pos")  # get all valid positives
            masked_anchor_positive_dists = anchor_positive_dists.squeeze(2).masked_fill(
                pos_mask == 0, float("-inf")
            )  # fill all invalid positives with inf so they are not considered in the min
            _, pos_max_indices = torch.max(masked_anchor_positive_dists, dim=1)
            # print(pos_max_indices)

            hard_mask = torch.zeros(len(labels), len(labels), len(labels))
            hard_mask[torch.arange(len(labels)), pos_max_indices, neg_min_indices] = 1
            hard_mask = hard_mask.to(mask.device)
            # combine with base mask
            mask = torch.logical_and(mask, hard_mask)

        elif (
            self.mode == "semi-hard"
        ):  # select the negatives with a bigger distance than the positive but a difference smaller than the margin
            semi_hard_mask = get_semi_hard_mask(labels, distance_matrix)
            # combine with base mask
            semi_hard_mask = semi_hard_mask.to(mask.device)
            mask = torch.logical_and(mask, semi_hard_mask)

        return mask.float()


class TripletLossOffline(nn.Module):
    """
    TripletLossOffline operates on triplets.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, labels: gtypes.MergedLabels, images: torch.Tensor = torch.Tensor()
    ) -> gtypes.LossPosNegDist:
        """
        Compute loss.

        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)

        Returns:
          Scalar loss value.
        """
        # NOTE(rob2u): custom implementation to return pos/neg distances.
        # Offline has 3 chunks, anchors, postives and negatives.
        third = embeddings.size()[0] // 3
        anchors, positives, negatives = embeddings[:third], embeddings[third : 2 * third], embeddings[2 * third :]

        distance_positive = torch.functional.norm(anchors - positives, dim=1)
        distance_negative = torch.functional.norm(anchors - negatives, dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin).mean()
        return losses.mean(), distance_positive.mean(), distance_negative.mean()


class TripletLossOfflineNative(nn.Module):
    """
    TripletLossOfflineNative is the native torch triplet loss implementation. It
    does not support positive, negative distance computation and fallbacks to -1.
    Use to validate custom implementation that exposes more metrics.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=margin)

    def forward(
        self, embeddings: torch.Tensor, labels: gtypes.MergedLabels, images: torch.Tensor = torch.Tensor()
    ) -> gtypes.LossPosNegDist:
        # Offline has 3 chunks, anchors, postives and negatives.
        third = embeddings.size()[0] // 3
        anchors, positives, negatives = embeddings[:third], embeddings[third : 2 * third], embeddings[2 * third :]
        NO_VALUE = torch.tensor([-1])
        return self.loss(anchors, positives, negatives), NO_VALUE, NO_VALUE


class L2SPRegularization_Wrapper(nn.Module):
    """Wrapper that adds L2SP regularization to any loss"""

    def __init__(
        self,
        loss: nn.Module,
        model: nn.Module,
        path_to_pretrained_weights: str,
        alpha: float,
        beta: float,
        log_func: Callable[[str, float], None] = lambda x, y: None,
    ):
        super().__init__()
        assert path_to_pretrained_weights is not None, "Path to pretrained weights must be provided"
        self.loss = loss
        self.model = model
        self.l2sp_loss = l2.L2_SP(model, path_to_pretrained_weights, alpha, beta)
        self.log = log_func

    def forward(self, *args: List[Any], **kwargs: Dict[str, Any]) -> gtypes.LossPosNegDist:
        standard_loss, anchor_positive_dist_mean, anchor_negative_dist_mean = self.loss(*args, **kwargs)
        l2sp_loss = self.l2sp_loss(self.model)
        if type(l2sp_loss) == torch.Tensor:
            self.log("l2_sp", l2sp_loss.item())
        else:
            self.log("l2_sp", l2sp_loss)
        return standard_loss + l2sp_loss, anchor_positive_dist_mean, anchor_negative_dist_mean


def get_loss(
    loss_mode: str,
    log_func: Callable[[str, float], None] = lambda x, y: None,
    **kw_args: Any,
) -> Callable[[torch.Tensor, gtypes.BatchLabel], gtypes.LossPosNegDist]:
    l2sp = False
    if "l2sp" in loss_mode:
        loss_mode = loss_mode.replace("/l2sp", "")
        l2sp = True

    loss_module: Union[torch.nn.Module, None] = None

    if loss_mode == "online/hard":
        loss_module = TripletLossOnline(mode="hard", margin=kw_args["margin"])
    elif loss_mode == "online/semi-hard":
        loss_module = TripletLossOnline(mode="semi-hard", margin=kw_args["margin"])
    elif loss_mode == "online/soft":
        loss_module = TripletLossOnline(mode="soft", margin=kw_args["margin"])
    elif loss_mode == "offline":
        loss_module = TripletLossOffline(margin=kw_args["margin"])
    elif loss_mode == "offline/native":
        loss_module = TripletLossOfflineNative(margin=kw_args["margin"])
    elif loss_mode == "softmax/arcface":
        loss_module = ArcFaceLoss(
            embedding_size=kw_args["embedding_size"],
            num_classes=kw_args["num_classes"],
            s=kw_args["s"],
            margin=kw_args["margin"],
            accelerator=kw_args["accelerator"],
        )
    elif loss_mode == "softmax/vpl":
        loss_module = VariationalPrototypeLearning(
            embedding_size=kw_args["embedding_size"],
            num_classes=kw_args["num_classes"],
            batch_size=kw_args["batch_size"],
            s=kw_args["s"],
            margin=kw_args["margin"],
            delta_t=kw_args["delta_t"],
            mem_bank_start_epoch=kw_args["mem_bank_start_epoch"],
            accelerator=kw_args["accelerator"],
        )
    elif loss_mode == "distillation/offline/response-based":
        loss_module = OfflineResponseBasedLoss(teacher_model_wandb_link=kw_args["teacher_model_wandb_link"])
    else:
        raise ValueError(f"Loss mode {loss_mode} not supported")

    if l2sp:
        return L2SPRegularization_Wrapper(
            loss=loss_module,
            model=kw_args["model"],
            path_to_pretrained_weights=kw_args["path_to_pretrained_weights"],
            alpha=kw_args["l2_alpha"],
            beta=kw_args["l2_beta"],
            log_func=log_func,
        )

    return loss_module


if __name__ == "__main__":
    # Test TripletLossOnline with example
    batch_size = 4
    embedding_dim = 2
    margin = 1.0
    triplet_loss = TripletLossOnline(margin=margin, mode="hard")
    triplet_loss_soft = TripletLossOnline(margin=margin, mode="soft")
    triplet_loss_semi_hard = TripletLossOnline(margin=margin, mode="semi-hard")
    embeddings = torch.tensor([[1.0], [0.5], [-1.0], [0.0]])
    labels = ["0", "0", "1", "1"]

    loss_013 = torch.relu(  # anchor 1.0 positive 0.5 negative 0.0
        torch.linalg.vector_norm(embeddings[0] - embeddings[1])
        - torch.linalg.vector_norm(embeddings[0] - embeddings[3])
        + margin
    )
    loss_012 = torch.relu(  # anchor 1.0 positive 0.5 negative -1.0
        torch.linalg.vector_norm(embeddings[0] - embeddings[1])
        - torch.linalg.vector_norm(embeddings[0] - embeddings[2])
        + margin
    )

    loss_103 = torch.relu(  # anchor 0.5 positive 1.0 negative 0.0
        torch.linalg.vector_norm(embeddings[1] - embeddings[0])
        - torch.linalg.vector_norm(embeddings[1] - embeddings[3])
        + margin
    )
    loss_102 = torch.relu(  # anchor 0.5 positive 1.0 negative -1.0
        torch.linalg.vector_norm(embeddings[1] - embeddings[0])
        - torch.linalg.vector_norm(embeddings[1] - embeddings[2])
        + margin
    )

    loss_231 = torch.relu(  # anchor -1.0 positive 0.0 negative 0.5
        torch.linalg.vector_norm(embeddings[2] - embeddings[3])
        - torch.linalg.vector_norm(embeddings[2] - embeddings[1])
        + margin
    )
    loss_230 = torch.relu(  # anchor -1.0 positive 0.0 negative 1.0
        torch.linalg.vector_norm(embeddings[2] - embeddings[3])
        - torch.linalg.vector_norm(embeddings[2] - embeddings[0])
        + margin
    )

    loss_321 = torch.relu(  # anchor 0.0 positive -1.0 negative 0.5
        torch.linalg.vector_norm(embeddings[3] - embeddings[2])
        - torch.linalg.vector_norm(embeddings[3] - embeddings[1])
        + margin
    )

    loss_manual = (loss_013 + loss_103 + loss_231 + loss_321) / 4
    loss = triplet_loss(embeddings, labels)
    loss_semi = triplet_loss_semi_hard(embeddings, labels)
    loss_semi_manual = (loss_013 + loss_012 + loss_102 + loss_230 + loss_231) / 5
    print(f"Correct Hard Loss {loss_manual}")
    print(f"Hard Loss {loss}")
    print(f"Correct Semi Hard Loss {loss_semi_manual}")
    print(f"Semi Hard Loss {loss_semi}")
