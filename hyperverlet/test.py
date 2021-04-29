import torch
from torch import nn
from torch.utils.data import DataLoader

from hyperverlet.timer import timer
from hyperverlet.utils.misc import send_to_device, torch_to_numpy


def test(solver: nn.Module, dataset, device, config):
    solver.eval()
    criterion = nn.MSELoss()

    train_args = config["train_args"]
    num_workers = train_args["num_workers"]

    loader = DataLoader(dataset, num_workers=num_workers, pin_memory=device.type == 'cuda', batch_size=len(dataset))

    with torch.no_grad():
        batch = next(iter(loader))

        q_base = batch['q'].transpose(0, 1).contiguous().to(device, non_blocking=True)
        p_base = batch['p'].transpose(0, 1).contiguous().to(device, non_blocking=True)
        mass = batch['mass'].to(device, non_blocking=True)
        trajectory = batch['trajectory'].transpose(0, 1).contiguous().to(device, non_blocking=True)
        extra_args = send_to_device(batch['extra_args'], device, non_blocking=True)

        (q, p), inference_time = timer(lambda: solver.trajectory(dataset.experiment, q_base[0], p_base[0], mass, trajectory, **extra_args), 'solving', return_time=True)

        q_loss, p_loss = criterion(q, q_base), criterion(p, p_base)
        print(f'final loss: {q_loss.item(), p_loss.item()}')

    return {
        "inference_time": inference_time,
        "q": q.cpu().detach().numpy(),
        "gt_q": q_base.cpu().detach().numpy(),
        "p": p.cpu().detach().numpy(),
        "gt_p": p_base.cpu().detach().numpy(),
        "mass": mass.cpu().detach().numpy(),
        "trajectory": trajectory.cpu().detach().numpy(),
        "extra_args": torch_to_numpy(send_to_device(extra_args, torch.device("cpu"), non_blocking=True))
    }