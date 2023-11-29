import torch
from torch.nn import TripletMarginLoss
from gorillatracker.triplet_loss import TripletLossOffline, TripletLossOnline


def calc_triplet_loss(anchor, positive, negative, margin):
    loss = torch.relu(torch.linalg.vector_norm(anchor - positive) - torch.linalg.vector_norm(anchor - negative) + margin)
    return loss

def approx_equal(a, b, eps=1e-6):
    return torch.abs(a - b) < eps

def test_tripletloss_offline():
    a, p, n = torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([-1.0])
    labels = torch.tensor([0, 0, 1])
    triplet_loss = TripletLossOffline(margin=1.0)
    sample_triplet = torch.stack([a, p, n])
    loss, d_p, d_n = triplet_loss(sample_triplet, labels)
    loss_manual = calc_triplet_loss(a, p, n, margin=1.0)
    assert loss == loss_manual and loss == 0.0
    assert d_p == torch.linalg.norm(a - p)
    assert d_n == torch.linalg.norm(a - n)
    
    a, p, n = torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.5])
    loss, d_p, d_n = triplet_loss(torch.stack([a, p, n]), labels)
    assert approx_equal(loss,calc_triplet_loss(a, p, n, margin=1.0)) and approx_equal(loss, 1.0)
    assert approx_equal(d_p, torch.linalg.norm(a - p))
    assert approx_equal(d_n, torch.linalg.norm(a - n))
    
    loss_torch = TripletMarginLoss(margin=1.0)(a, p, n)
    assert approx_equal(loss_torch, loss) and approx_equal(loss_torch, 1.0)


def test_tripletloss_online_soft():
    pass

def test_tripletloss_online_hard():
    pass

def test_tripletloss_online_semi_hard():
    pass

if __name__ == "__main__":
    test_tripletloss_offline()
    test_tripletloss_online_soft()
    test_tripletloss_online_hard()
    test_tripletloss_online_semi_hard()
