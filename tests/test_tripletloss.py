import torch
from gorillatracker.triplet_loss import TripletLossOffline, TripletLossOnline


def calc_triplet_loss(anchor, positive, negative, margin):
    loss = torch.relu(torch.linalg.vector_norm(anchor - positive) - torch.linalg.vector_norm(anchor - negative) + margin)
    return loss

def test_tripletloss_offline():
    a, p, n = torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([-1.0])
    triplet_loss = TripletLossOffline(margin=1.0)
    sample_triplet = torch.stack([a, p, n])
    loss = triplet_loss(sample_triplet)
    loss_manual = calc_triplet_loss(a, p, n, margin=1.0)
    assert loss == loss_manual
    
    a, p, n = torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.5])
    assert triplet_loss(torch.stack([a, p, n])) == calc_triplet_loss(a, p, n, margin=1.0)


def test_tripletloss_online_soft():
    pass

def test_tripletloss_online_hard():
    pass

def test_tripletloss_online_semi_hard():
    pass

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
