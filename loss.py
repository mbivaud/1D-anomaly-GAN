import torch


def residual_loss(x, g_z):
    return torch.sum(torch.abs(x - g_z))


def discrimination_loss(x, z, D, G):
    feature_G_z, _ = D(G(z))
    feature_x, _ = D(x)
    return torch.sum(torch.abs(feature_x - feature_G_z))


def anomaly_score(r_loss, d_loss, __lambda__=0.1):
    return (1 - __lambda__) * r_loss + __lambda__ * d_loss

