import random

from torch import optim, nn
from tqdm import trange


def train(solver, experiment, q_base, p_base, trajectory, trajectory_fitting=True):
    assert solver.trainable, 'Solver is not trainable'

    optimizer = optim.AdamW(solver.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for iteration in trange(100000, desc='Training iteration'):
        optimizer.zero_grad(set_to_none=True)

        batch_size = 4
        if iteration < -1:
            start = 0
        else:
            start = random.randint(0, trajectory.size(0) - batch_size - 1)

        end = start + batch_size

        if trajectory_fitting:
            q, p = solver.trajectory(experiment, q_base[start], p_base[start], experiment.mass, trajectory[start:end + 1], disable_print=True)
            loss = criterion(q, q_base[start:end + 1]) + criterion(p, p_base[start:end + 1])
        else:
            loss = solver.loss(experiment, q_base[start:end + 1], p_base[start:end + 1], experiment.mass, trajectory[start:end + 1])

        loss.backward()
        nn.utils.clip_grad_norm_(solver.parameters(), 1)
        optimizer.step()

        if iteration % 100 == 0:
            print(f'loss: {loss.item()}')
