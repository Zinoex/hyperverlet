import random

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from hyperverlet.utils.misc import send_to_device


def train(solver, dataset, device, config):
    assert solver.trainable, 'Solver is not trainable'

    train_args = config["train_args"]
    epochs = train_args["epoch"]
    batch_size = train_args["batch_size"]
    num_workers = train_args["num_workers"]

    optimizer = optim.AdamW(solver.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(dataset, num_workers=num_workers, pin_memory=device.type == 'cuda', batch_size=batch_size, shuffle=True)

    for epoch in trange(epochs, desc='Epoch'):
        for iteration, batch in enumerate(tqdm(loader, desc='Training iteration')):
            optimizer.zero_grad(set_to_none=True)

            q_base = batch['q'].to(device, non_blocking=True).transpose_(0, 1)
            p_base = batch['p'].to(device, non_blocking=True).transpose_(0, 1)
            mass = batch['mass'].to(device, non_blocking=True)
            trajectory = batch['trajectory'].to(device, non_blocking=True).transpose_(0, 1)
            extra_args = send_to_device(batch['extra_args'], device, non_blocking=True)

            # q, p = solver.trajectory(dataset.experiment, q_base[0], p_base[0], mass, trajectory, disable_print=True, **extra_args)
            #loss = criterion(q, q_base) + criterion(p, p_base)

            q_base_res, p_base_res = solver.get_base_residuals(dataset.experiment, q_base, p_base, mass, trajectory, disable_print=True, **extra_args)
            q_hyper_res, p_hyper_res = solver.get_hyper_residuals(dataset.experiment, q_base, p_base, mass, trajectory, disable_print=True, **extra_args)

            loss = criterion(q_hyper_res, q_base_res) + criterion(p_base_res, p_base_res)

            loss.backward()
            # nn.utils.clip_grad_norm_(solver.parameters(), 1)
            optimizer.step()

            if iteration % 100 == 0:
                print(f'loss: {loss.item()}')
