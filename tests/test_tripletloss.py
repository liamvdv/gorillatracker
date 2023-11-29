import torch
from torch.nn import TripletMarginLoss
from gorillatracker.triplet_loss import TripletLossOffline, TripletLossOnline


def calc_loss_of_triplet(anchor, positive, negative, margin):
    loss = torch.relu(torch.linalg.vector_norm(anchor - positive) - torch.linalg.vector_norm(anchor - negative) + margin)
    return loss


def calc_loss_and_distance_of_triplet(anchor, positive, negative, margin):
    loss = torch.relu(torch.linalg.vector_norm(anchor - positive) - torch.linalg.vector_norm(anchor - negative) + margin)
    return loss, torch.linalg.vector_norm(anchor - positive), torch.linalg.vector_norm(anchor - negative)

def approx_equal(a, b, eps=1e-6):
    return torch.abs(a - b) < eps

def test_tripletloss_offline():
    a, p, n = torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([-1.0])
    labels = torch.tensor([0, 0, 1])
    triplet_loss = TripletLossOffline(margin=1.0)
    sample_triplet = torch.stack([a, p, n])
    loss, d_p, d_n = triplet_loss(sample_triplet, labels)
    loss_manual = calc_loss_of_triplet(a, p, n, margin=1.0)
    assert loss == loss_manual and loss == 0.0
    assert d_p == torch.linalg.norm(a - p)
    assert d_n == torch.linalg.norm(a - n)
    
    a, p, n = torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.5])
    loss, d_p, d_n = triplet_loss(torch.stack([a, p, n]), labels)
    assert approx_equal(loss,calc_loss_of_triplet(a, p, n, margin=1.0)) and approx_equal(loss, 1.0)
    assert approx_equal(d_p, torch.linalg.norm(a - p))
    assert approx_equal(d_n, torch.linalg.norm(a - n))
    
    loss_torch = TripletMarginLoss(margin=1.0)(a, p, n)
    assert approx_equal(loss_torch, loss) and approx_equal(loss_torch, 1.0)


def test_tripletloss_online_soft():
    a, ap, n, np = torch.tensor([3.0]), torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([-0.5])
    loss012, d_p012, d_n012 = calc_loss_and_distance_of_triplet(a, ap, n, margin=1.0)
    loss013, d_p013, d_n013 = calc_loss_and_distance_of_triplet(a, ap, np, margin=1.0)
    loss102, d_p102, d_n102 = calc_loss_and_distance_of_triplet(ap, a, n, margin=1.0)
    loss103, d_p103, d_n103 = calc_loss_and_distance_of_triplet(ap, a, np, margin=1.0)
    
    loss230, d_p230, d_n230 = calc_loss_and_distance_of_triplet(n, np, a, margin=1.0)
    loss231, d_p231, d_n231 = calc_loss_and_distance_of_triplet(n, np, ap, margin=1.0)
    loss320, d_p320, d_n320 = calc_loss_and_distance_of_triplet(np, n, a, margin=1.0)
    loss321, d_p321, d_n321 = calc_loss_and_distance_of_triplet(np, n, ap, margin=1.0)
    
    loss_manual = (loss012 + loss013 + loss102 + loss103 + loss230 + loss231 + loss320 + loss321) / 8
    distance_positive_manual = (d_p012 + d_p013 + d_p102 + d_p103 + d_p230 + d_p231 + d_p320 + d_p321) / 8
    distance_negative_manual = (d_n012 + d_n013 + d_n102 + d_n103 + d_n230 + d_n231 + d_n320 + d_n321) / 8
    
    loss_online_soft, distance_positive_online_soft, distance_negative_online_soft  = TripletLossOnline(margin=1.0, mode="soft")(torch.stack([a, ap, n, np]), torch.tensor([0, 0, 1, 1]))
    
    assert approx_equal(loss_manual, loss_online_soft)
    assert approx_equal(distance_positive_manual, distance_positive_online_soft)
    assert approx_equal(distance_negative_manual, distance_negative_online_soft)
    

def test_tripletloss_online_hard():
    pass

def test_tripletloss_online_semi_hard():
    pass

if __name__ == "__main__":
    test_tripletloss_offline()
    test_tripletloss_online_soft()
    test_tripletloss_online_hard()
    test_tripletloss_online_semi_hard()
