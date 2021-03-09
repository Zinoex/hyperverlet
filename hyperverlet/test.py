import torch
from torch import nn

from hyperverlet.timer import timer


def test(solver, experiment, trajectory, q_base, p_base):
    criterion = nn.MSELoss()

    with torch.no_grad():
        q, p = timer(lambda: solver.trajectory(experiment, experiment.q0, experiment.p0, experiment.mass, trajectory), 'solving')

        q_loss, p_loss = criterion(q, q_base), criterion(p, p_base)
        print(f'final loss: {q_loss.item(), p_loss.item()}')

    return q, p
