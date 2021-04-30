from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from hyperverlet.loss import TimeDecayMSELoss
from hyperverlet.utils.misc import send_to_device


def train(solver: nn.Module, dataset, device, config):
    assert solver.trainable, 'Solver is not trainable'
    solver.train()

    train_args = config["train_args"]
    epochs = train_args["epoch"]
    loss_method = train_args["loss"]
    batch_size = train_args["batch_size"]
    num_workers = train_args["num_workers"]
    time_decay = train_args["time_decay"]

    assert loss_method in ['phase_space', 'energy', "residual"]

    optimizer = optim.AdamW(solver.parameters(), lr=1e-3)
    criterion = TimeDecayMSELoss(time_decay)

    loader = DataLoader(dataset, num_workers=num_workers, pin_memory=device.type == 'cuda', batch_size=batch_size, shuffle=True)
    data_len = len(loader)
    summary_writer = SummaryWriter()

    for epoch in trange(epochs, desc='Epoch'):
        for iteration, batch in enumerate(tqdm(loader, desc='Training iteration')):
            optimizer.zero_grad(set_to_none=True)

            q_base = batch['q'].transpose(0, 1).contiguous().to(device, non_blocking=True)
            p_base = batch['p'].transpose(0, 1).contiguous().to(device, non_blocking=True)
            mass = batch['mass'].to(device, non_blocking=True)
            trajectory = batch['trajectory'].transpose(0, 1).contiguous().to(device, non_blocking=True)
            extra_args = send_to_device(batch['extra_args'], device, non_blocking=True)

            if loss_method in ["phase_space", "energy"]:
                q, p = solver.trajectory(dataset.experiment, q_base[0], p_base[0], mass, trajectory, disable_print=True, **extra_args)

            if loss_method == "phase_space":
                loss = criterion(q, q_base) + criterion(p, p_base)
            elif loss_method == "energy":
                gt_ke = dataset.experiment.energy.kinetic_energy(mass, p_base, **extra_args)
                gt_pe = dataset.experiment.energy.potential_energy(mass, q_base, **extra_args)

                pred_ke = dataset.experiment.energy.kinetic_energy(mass, p, **extra_args)
                pred_pe = dataset.experiment.energy.potential_energy(mass, q, **extra_args)

                loss = criterion(gt_ke, pred_ke) + criterion(gt_pe, pred_pe)
            elif loss_method == "residual":
                q_res, p_res = solver.get_residuals(dataset.experiment, q_base, p_base, mass, trajectory, disable_print=True, **extra_args)
                q_hyper, p_hyper = solver.hyper_trajectory(dataset.experiment, q_base, p_base, mass, trajectory, disable_print=True, **extra_args)

                loss = criterion(q_hyper, q_res) + criterion(p_hyper, p_res)
            else:
                raise NotImplementedError()

            loss.backward()
            optimizer.step()

            loss = loss.item()
            summary_writer.add_scalar('loss/{}'.format(loss_method), loss, global_step=epoch * data_len + iteration)

            if iteration % 100 == 0:
                print(f'loss: {loss}')

        # Log weights to Tensorboard
        for name, param in solver.hypersolver.named_parameters():
            summary_writer.add_histogram('weights/{}'.format(name), param, epoch)
