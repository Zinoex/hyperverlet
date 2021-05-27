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


def qp_mean(q, p, gt_q, gt_p):
    q_loss, p_loss = qp_loss(q, p, gt_q, gt_p)
    return (q_loss + p_loss) / 2


def print_qp_mean_loss(q, p, gt_q, gt_p, label='final loss'):
    qp_mean_loss = qp_mean(q, p, gt_q, gt_p)
    print("{:.3e} {}".format(qp_mean_loss, label))


def valid_prediction_time(q, p, gt_q, gt_p, trajectory, threshold=0.006):
    if torch.is_tensor(q):
        z = torch.cat([q, p], dim=-1)
        gt_z = torch.cat([gt_q, gt_p], dim=-1)

        num_config = z.size(1)

        squared_diff = (z - gt_z) ** 2

        mean_axis = tuple(range(2, squared_diff.dim()))
        z_loss = squared_diff.mean(dim=mean_axis).sqrt()

        mask = torch.cat([z_loss > threshold, torch.full((1, num_config), True)])

        tidx = torch.argmax(mask, dim=0) - 1
        cidx = torch.arange(num_config)
        return trajectory[tidx, cidx].mean().item()

    else:
        z = np.concatenate([q, p], axis=-1)
        gt_z = np.concatenate([gt_q, gt_p], axis=-1)

        num_config = z.shape[1]

        squared_diff = (z - gt_z) ** 2

        mean_axis = tuple(range(2, squared_diff.ndim))
        z_loss = np.sqrt(squared_diff.mean(axis=mean_axis))

        mask = np.concatenate([z_loss > threshold, np.full((1, num_config), True)])

        tidx = np.argmax(mask, axis=0) - 1
        cidx = np.arange(num_config)
        return trajectory[tidx, cidx].mean()


def print_valid_prediction_time(q, p, gt_q, gt_p, trajectory, threshold=0.1, label='vpt'):
    vpt = valid_prediction_time(q, p, gt_q, gt_p, trajectory, threshold=threshold)

    print("{} {}".format(round(vpt), label))
