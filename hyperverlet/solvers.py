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

        for i, (t, dt) in enumerate(zip(trajectory[:-1], dtrajectory)):
            q, p = self(func, q, p, m, t, dt, **kwargs)
            q_traj[i + 1], p_traj[i + 1] = q, p

        return q_traj, p_traj


class HyperSolverMixin:
    def residual_trajectory(self, func: Callable, q_base: torch.Tensor, p_base: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, **kwargs):
        q_traj = torch.zeros((trajectory.size(0) - 1, *q_base.shape[1:]), dtype=q_base.dtype, device=q_base.device)
        p_traj = torch.zeros((trajectory.size(0) - 1, *p_base.shape[1:]), dtype=p_base.dtype, device=p_base.device)

        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(trajectory[:-1], dtrajectory)):
            q_res, p_res = self.residual(func, q_base[i], q_base[i + 1], p_base[i], p_base[i + 1], m, t, dt, **kwargs)
            q_traj[i], p_traj[i] = q_res, p_res

        return q_traj, p_traj

    def residual(self, func: Callable, q: torch.Tensor, p: torch.Tensor, q_next: torch.Tensor, p_next: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        raise NotImplemented()

    def hypersolver_trajectory(self, solver: Callable, func: Callable, q_base: torch.Tensor, p_base: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, **kwargs):
        q_traj = torch.zeros((trajectory.size(0) - 1, *q_base.shape[1:]), dtype=q_base.dtype, device=q_base.device)
        p_traj = torch.zeros((trajectory.size(0) - 1, *p_base.shape[1:]), dtype=p_base.dtype, device=p_base.device)

        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(trajectory[:-1], dtrajectory)):
            dq, dp = func(q_base[i], p_base[i], m, t)
            q, p = solver(q_base[i], p_base[i], dq, dp, m, t, dt, **kwargs)
            q_traj[i], p_traj[i] = q, p

        return q_traj, p_traj


class EulerSolver(BaseSolver):
    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t)    # t1 --LAMMPSx2--> t2

        return q + dq * dt, p + dp * dt


class HyperEulerSolver(BaseSolver, HyperSolverMixin):
    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t)
        hq, hp = self.hypersolver(q, p, dq, dp, m, t, dt)

        return q + dq * dt + hq * (dt ** 2), p + dp * dt + hp * (dt ** 2)

    def residual(self, func: Callable, q: torch.Tensor, q_next: torch.Tensor, p: torch.Tensor, p_next: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t)

        return (q - q_next - dq * dt) / (dt ** 2), (p - p_next - dp * dt) / (dt ** 2)

    def loss(self, func, q_base: torch.Tensor, p_base: torch.Tensor, m: torch.Tensor, trajectory, **kwargs):
        q_res, p_res = self.residual_trajectory(func, q_base, p_base, m, trajectory, **kwargs)
        q_model, p_model = self.hypersolver_trajectory(self.hypersolver, func, q_base, p_base, m, trajectory, **kwargs)

        return torch.mean((q_res - q_model) ** 2) + torch.mean((p_res - p_model) ** 2)


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
        dq, dp = func(q, p, m, t)
        hq, hp = self.hypersolver_q(q, p, dq, dp, m, t, dt)
        q_next = q + p * dt / m + (dp / (2 * m)) * (dt ** 2) + hq * (dt ** 3)

        dq_next, dp_next = func(q_next, p, m, t)
        hq, hp = self.hypersolver_p(q_next, p, dq_next, dp_next, m, t, dt)
        p_next = p + ((dp + dp_next) / 2) * dt + hp * (dt ** 3)

        return q_next, p_next
