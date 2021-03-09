import random

from torch import optim, nn
from tqdm import trange


def train(solver, experiment, q_base, p_base, trajectory, trajectory_fitting=True):
    assert solver.trainable, 'Solver is not trainable'

    optimizer = optim.AdamW(solver.parameters(), lr=0.5 * 1e-2)
    criterion = nn.MSELoss()

    for iteration in trange(1000, desc='Training iteration'):
        optimizer.zero_grad(set_to_none=True)

        batch_size = 4
        if iteration < -1:
            start = 0
        else:
            start = random.randint(0, trajectory.size(0) - batch_size)

        end = start + batch_size

        if trajectory_fitting:
            q, p = solver.trajectory(experiment, q_base[start], p_base[start], experiment.mass, trajectory[start:end])
            loss = criterion(q, q_base[start:end]) + criterion(p, p_base[start:end])
        else:
            loss = solver.loss(experiment, q_base[start:end], p_base[start:end], experiment.mass, trajectory[start:end])

        loss.backward()
        optimizer.step()

        print(f'loss: {loss.item()}')
