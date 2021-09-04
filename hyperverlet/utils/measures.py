import numpy as np
import torch


def qp_loss(q, p, gt_q, gt_p, dim=None):
    if torch.is_tensor(q):
        def mse(pred, target):
            return torch.mean((pred - target) ** 2, dim=dim)

        q_loss, p_loss = mse(q, gt_q).item(), mse(p, gt_p).item()
    else:
        def mse(pred, target):
            return np.mean((pred - target) ** 2, axis=dim)
        q_loss, p_loss = mse(q, gt_q), mse(p, gt_p)

    return q_loss, p_loss


def qp_mean(q, p, gt_q, gt_p, dim=None):
    q_loss, p_loss = qp_loss(q, p, gt_q, gt_p, dim=dim)

    return (q_loss + p_loss) / 2


def z_loss(q, p, gt_q, gt_p):
    q, p, gt_q, gt_p = (torch.as_tensor(x) for x in [q, p, gt_q, gt_p])

    z = torch.cat([q, p], dim=-1).view(q.size(0), q.size(1), -1)
    gt_z = torch.cat([gt_q, gt_p], dim=-1).view(q.size(0), q.size(1), -1)

    return torch.mean((z - gt_z) ** 2, dim=(0, 2))


def z_loss_dist(q, p, gt_q, gt_p):
    z_mse = z_loss(q, p, gt_q, gt_p)
    z_mean, z_std = z_mse.mean(), z_mse.std(0)

    return z_mean, z_std


def print_z_loss(q, p, gt_q, gt_p, label='final loss'):
    z_mean, z_std = z_loss_dist(q, p, gt_q, gt_p)

    print("{:.3e}\\pm {:.3e} {}".format(z_mean, z_std, label))
