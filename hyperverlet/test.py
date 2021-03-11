import torch
from torch import nn

from hyperverlet.timer import timer
from hyperverlet.utils import send_to_device


def test(solver, dataset, device):
    criterion = nn.MSELoss()

    with torch.no_grad():
        batch = dataset[0]

        q_base = batch['q'].to(device, non_blocking=True)
        p_base = batch['p'].to(device, non_blocking=True)
        mass = batch['mass'].to(device, non_blocking=True)
        trajectory = batch['trajectory'].to(device, non_blocking=True)
        extra_args = send_to_device(batch['extra_args'], device, non_blocking=True)

        q, p = timer(lambda: solver.trajectory(dataset.experiment, q_base[0], p_base[0], mass, trajectory, **extra_args), 'solving')

        q_loss, p_loss = criterion(q, q_base), criterion(p, p_base)
        print(f'final loss: {q_loss.item(), p_loss.item()}')

    return q, q_base, p, p_base, mass, trajectory, extra_args
