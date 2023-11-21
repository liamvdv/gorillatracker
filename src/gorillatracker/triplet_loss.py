from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

eps = 1e-16  # an arbitrary small value to be used for numerical stability tricks


def get_triplet_mask(labels):
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
    indices_equal = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
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
    # shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2).repeat(1, 1, batch_size)
    # shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1).repeat(1, batch_size, 1)
    # shape: (batch_size, batch_size, batch_size)

    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # step 3 - combine two masks
    mask = torch.logical_and(distinct_indices, valid_indices)

    return mask


def get_distance_mask(labels, valid: Literal["pos", "neg"] = "neg"):
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

    batch_size = labels.size()[0]
    indices_equal = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)

    if valid == "pos":
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = torch.logical_and(labels_equal, indices_not_equal)
    elif valid == "neg":
        labels_not_equal = labels.unsqueeze(0) != labels.unsqueeze(1)
        mask = torch.logical_and(labels_not_equal, indices_not_equal)

    return mask


def get_semi_hard_mask(
    labels,
    distance_matrix,
    margin=1.0,
):
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
    batch_size = labels.size()[0]
    indices_equal = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_not_equal = labels.unsqueeze(0) != labels.unsqueeze(1)

    distance_matrix_pos = distance_matrix * torch.logical_and(labels_equal, indices_not_equal).float()
    distance_matrix_neg = distance_matrix * torch.logical_and(labels_not_equal, indices_not_equal).float()

    # filter out all points where the distance to a negative is smaller than the max distance to a positive
    distance_difference = distance_matrix_pos.unsqueeze(2).repeat(1, 1, batch_size) - distance_matrix_neg.unsqueeze(
        1
    ).repeat(
        1, batch_size, 1
    )  # shape: (anchor: batch_size,positive: batch_size, negative: batch_size)

    # filter out all points where the distance to a negative is smaller than the max distance to a positive

    distance_difference = torch.nn.functional.relu(
        distance_difference
    )  # now only the triplets where dist_pos < dist_neg are left
    mask = get_triplet_mask(labels)
    semi_hard_mask = distance_matrix > 0.0
    semi_hard_mask = semi_hard_mask.to(mask.device)

    return torch.logical_and(mask, semi_hard_mask)

    # mask = torch.logical_not(get_triplet_mask(labels))
    # distance_matrix = distance_matrix + (mask.float() * (1 / eps))

    # print(distance_matrix)
    # # get the minimum for each anchor positive pair
    # min_distance_anchor_neg, min_distance_indices_anchor_neg = torch.min(
    #     distance_matrix, dim=2
    # )  # take the hardest negative for anchor positive pair
    # # get the minimum for each anchor
    # _, min_distance_indices_anchor = torch.min(min_distance_anchor_neg, dim=1)  # take the hardest negative for an anchor itself

    # hardest_neg_mask = torch.zeros(len(labels), len(labels), len(labels))
    # hardest_neg_mask[
    #     torch.arange(len(labels)),
    #     min_distance_indices_anchor,
    #     min_distance_indices_anchor_neg[torch.arange(len(labels)), min_distance_indices_anchor],
    # ] = 1

    # print(hardest_neg_mask)
    # hardest_neg_mask = hardest_neg_mask.to(mask.device)
    # # combine with base mask
    # return torch.logical_and(hardest_neg_mask, mask)


def euclidean_distance_matrix(embeddings):
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
    mask = (distance_matrix == 0.0).float()

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

    def __init__(self, margin=1.0, mode: Literal["hard", "semi-hard", "soft"] = "semi-hard"):
        super().__init__()
        self.margin = margin
        self.mode = mode

    def forward(self, embeddings, labels):
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
        triplet_loss *= mask

        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging
        num_losses = torch.sum(mask)
        triplet_loss = triplet_loss.sum() / (num_losses + eps)

        # TODO(rob2u): implement positive and negative distance means
        todo = torch.tensor(-1, dtype=torch.float32, device=triplet_loss.device)
        return triplet_loss, todo, todo

    def get_mask(self, distance_matrix, anchor_positive_dists, anchor_negative_dists, labels):
        mask = get_triplet_mask(labels)

        if self.mode == "hard":  # take only the hardest negative as a negative per anchor
            neg_mask = get_distance_mask(labels, valid="neg")  # get all valid negatives

            # for each anchor compute the min distance to a negative
            _, neg_min_indices = torch.min(
                anchor_negative_dists.squeeze(1) + (((1.0 - neg_mask.int()) * (1 / eps))), dim=1
            )  # TODO find better solution for this

            hard_mask = torch.zeros(len(labels), len(labels), len(labels))
            hard_mask[torch.arange(len(labels)), :, neg_min_indices] = 1
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

        mask = mask.float()
        mask = mask.to(labels.device)
        return mask


class TripletLossOffline(nn.Module):
    """
    TripletLossOffline operates on triplets.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
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


def get_triplet_loss(loss_mode: str, margin: float):
    loss_modes = {
        "online/hard": TripletLossOnline(margin=margin, mode="hard"),
        "online/semi-hard": TripletLossOnline(margin=margin, mode="semi-hard"),
        "online/soft": TripletLossOnline(margin=margin, mode="soft"),
        "offline": TripletLossOffline(margin=margin),
    }
    return loss_modes[loss_mode]


if __name__ == "__main__":
    # Test TripletLossOnline with example
    batch_size = 4
    embedding_dim = 2
    margin = 1.0
    triplet_loss = TripletLossOnline(margin=margin, mode="hard")
    triplet_loss_soft = TripletLossOnline(margin=margin, mode="soft")
    triplet_loss_semi_hard = TripletLossOnline(margin=margin, mode="semi-hard")
    embeddings = torch.tensor([[1.0], [0.5], [-1.0], [0.0]])
    labels = torch.tensor([0, 0, 1, 1])

    loss_manual_1 = torch.relu(  # anchor 1.0 positive 0.5 negative 0.0
        torch.linalg.vector_norm(embeddings[0] - embeddings[1])
        - torch.linalg.vector_norm(embeddings[0] - embeddings[3])
        + margin
    )
    loss_manual_2 = torch.relu(  # anchor 0.5 positive 1.0 negative 0.0
        torch.linalg.vector_norm(embeddings[1] - embeddings[0])
        - torch.linalg.vector_norm(embeddings[1] - embeddings[3])
        + margin
    )
    loss_manual_3 = torch.relu(  # anchor -1.0 positive 0.0 negative 0.5
        torch.linalg.vector_norm(embeddings[2] - embeddings[3])
        - torch.linalg.vector_norm(embeddings[2] - embeddings[1])
        + margin
    )
    loss_manual_4 = torch.relu(  # anchor 0.0 positive -1.0 negative 0.5
        torch.linalg.vector_norm(embeddings[3] - embeddings[2])
        - torch.linalg.vector_norm(embeddings[3] - embeddings[1])
        + margin
    )
    loss_manual = (loss_manual_1 + loss_manual_2 + loss_manual_3 + loss_manual_4) / 4
    loss = triplet_loss(embeddings, labels)
    loss_semi = triplet_loss_semi_hard(embeddings, labels)
    loss_semi_manual = (loss_manual_1 + loss_manual_3) / 4
    print(f"Correct Hard Loss {loss_manual}")
    print(f"Hard Loss {loss}")
    print(f"Correct Semi Hard Loss {loss_semi_manual}")
    print(f"Semi Hard Loss {loss_semi}")
