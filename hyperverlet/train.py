import random

from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from hyperverlet.utils import send_to_device


def train(solver, dataset, device, trajectory_fitting=True):
    assert solver.trainable, 'Solver is not trainable'

    optimizer = optim.AdamW(solver.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    loader = DataLoader(dataset, num_workers=0, pin_memory=True, batch_size=8, shuffle=True)

    for epoch in trange(100, desc='Epoch'):
        for iteration, batch in enumerate(tqdm(loader, desc='Training iteration')):
            optimizer.zero_grad(set_to_none=True)

            q_base = batch['q'].to(device, non_blocking=True).transpose_(0, 1)
            p_base = batch['p'].to(device, non_blocking=True).transpose_(0, 1)
            mass = batch['mass'].to(device, non_blocking=True)
            trajectory = batch['trajectory'].to(device, non_blocking=True).transpose_(0, 1)
            extra_args = send_to_device(batch['extra_args'], device, non_blocking=True)

            if trajectory_fitting:
                q, p = solver.trajectory(dataset.experiment, q_base[0], p_base[0], mass, trajectory, disable_print=True, **extra_args)
                loss = criterion(q, q_base) + criterion(p, p_base)
            else:
                loss = solver.loss(dataset.experiment, q_base, p_base, mass, trajectory, **extra_args)

            loss.backward()
            nn.utils.clip_grad_norm_(solver.parameters(), 1)
            optimizer.step()

            if iteration % 100 == 0:
                print(f'loss: {loss.item()}')
