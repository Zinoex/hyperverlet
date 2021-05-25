import numpy as np
import torch
from torch import nn


def qp_loss(q, p, gt_q, gt_p):
    if torch.is_tensor(q):
        criterion = nn.MSELoss()

        q_loss, p_loss = criterion(q, gt_q).item(), criterion(p, gt_p).item()
    else:
        def mse(pred, target):
            return np.mean((pred - target) ** 2)
        q_loss, p_loss = mse(q, gt_q), mse(p, gt_p)

    return q_loss, p_loss


def print_qp_loss(q, p, gt_q, gt_p, label='final loss'):
    q_loss, p_loss = qp_loss(q, p, gt_q, gt_p)

    print("{:.3e} {}".format((q_loss + p_loss) / 2, label))


def valid_prediction_time(q, p, gt_q, gt_p, threshold=0.01):
    if torch.is_tensor(q):
        z = torch.cat([q, p], dim=-1)
        gt_z = torch.cat([gt_q, gt_p], dim=-1)

        squared_diff = (z - gt_z) ** 2

        mean_axis = tuple(range(1, squared_diff.dim()))
        z_loss = squared_diff.mean(dim=mean_axis).sqrt()

        return torch.where(z_loss > threshold)[0][0].item()
    else:
        z = np.concatenate([q, p], axis=-1)
        gt_z = np.concatenate([gt_q, gt_p], axis=-1)

        squared_diff = (z - gt_z) ** 2

        mean_axis = tuple(range(1, squared_diff.ndim))
        z_loss = np.sqrt(squared_diff.mean(axis=mean_axis))

        return np.where(z_loss > threshold)[0][0]


def print_valid_prediction_time(q, p, gt_q, gt_p, threshold=0.01, label='vpt'):
    vpt = valid_prediction_time(q, p, gt_q, gt_p, threshold=threshold)

    print("{:.3e} {}".format(vpt, label))
