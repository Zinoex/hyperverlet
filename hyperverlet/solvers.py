from typing import Callable

import torch
from torch import nn


class BaseSolver(nn.Module):

    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        raise NotImplemented()

    def trajectory(self, func: Callable, q0: torch.Tensor, p0: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, **kwargs):
        q_traj = torch.zeros((trajectory.size(0), *q0.size()), dtype=q0.dtype, device=q0.device)
        p_traj = torch.zeros((trajectory.size(0), *p0.size()), dtype=p0.dtype, device=p0.device)

        q_traj[0], p_traj[0] = q0, p0
        q, p = q0, p0

        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(trajectory[1:], dtrajectory)):
            q, p = self(func, q, p, m, t, dt, **kwargs)
            q_traj[i + 1], p_traj[i + 1] = q, p

        return q_traj, p_traj


class EulerSolver(BaseSolver):
    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t)    # t1 --LAMMPSx2--> t2

        return q + dq * dt, p + dp * dt


class HyperEulerSolver(BaseSolver):
    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t)
        hq, hp = self.hypersolver(q, p, m, t, dt)

        return q + dq * dt + hq * (dt ** 2), p + dp * dt + hp * (dt ** 2)


class VelocityVerletSolver(BaseSolver):
    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        _, dp = func(q, p, m, t)
        q_next = q + p * dt / m + (dp / (2 * m)) * (dt ** 2)

        _, dp_next = func(q_next, p, m, t)
        p_next = p + ((dp + dp_next) / 2) * dt

        return q_next, p_next


class HyperVelocityVerletSolver(BaseSolver):
    def __init__(self, hypersolver_q, hypersolver_p):
        super().__init__()

        self.hypersolver_q = hypersolver_q
        self.hypersolver_p = hypersolver_p

    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        double_mass = 2 * m

        _, dp = func(q, p, m, t)
        hq = self.hypersolver_q(q, p, m, t, dt)
        q_next = q + p * dt / m + (dp / (2 * m)) * (dt ** 2) + hq * (dt ** 2)

        dq_next, dp_next = func(q_next, p, m, t)
        hp = self.hypersolver_p(q_next, p, m, t, dt)
        p_next = p + ((dp + dp_next) / 2) * dt + hp * (dt ** 2)

        return q_next, p_next
