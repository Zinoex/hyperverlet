from typing import Callable

import torch
from torch import nn
from tqdm import tqdm


class BaseSolver(nn.Module):
    trainable = False

    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        raise NotImplemented()

    def trajectory(self, func: Callable, q0: torch.Tensor, p0: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = torch.zeros((trajectory.size(0), *q0.size()), dtype=q0.dtype, device=q0.device)
        p_traj = torch.zeros((trajectory.size(0), *p0.size()), dtype=p0.dtype, device=p0.device)

        q_traj[0], p_traj[0] = q0, p0
        q, p = q0, p0

        trajectory = trajectory.unsqueeze(-1)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = q.detach(), p.detach()
            q, p = self(func, q, p, m, t, dt, **kwargs)
            q_traj[i + 1], p_traj[i + 1] = q, p

        return q_traj, p_traj


class HyperSolverMixin:
    def residual_trajectory(self, func: Callable, q_base: torch.Tensor, p_base: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, **kwargs):
        q_traj = torch.zeros((trajectory.size(0) - 1, *q_base.shape[1:]), dtype=q_base.dtype, device=q_base.device)
        p_traj = torch.zeros((trajectory.size(0) - 1, *p_base.shape[1:]), dtype=p_base.dtype, device=p_base.device)

        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(trajectory[:-1], dtrajectory)):
            q_res, p_res = self.residual(func, q_base[i].detach(), q_base[i + 1].detach(), p_base[i], p_base[i + 1], m, t, dt, **kwargs)
            q_traj[i], p_traj[i] = q_res, p_res

        return q_traj, p_traj

    def residual(self, func: Callable, q: torch.Tensor, p: torch.Tensor, q_next: torch.Tensor, p_next: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        raise NotImplemented()

    def hypersolver_trajectory(self, solver: Callable, func: Callable, q_base: torch.Tensor, p_base: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, **kwargs):
        q_traj = torch.zeros((trajectory.size(0) - 1, *q_base.shape[1:]), dtype=q_base.dtype, device=q_base.device)
        p_traj = torch.zeros((trajectory.size(0) - 1, *p_base.shape[1:]), dtype=p_base.dtype, device=p_base.device)

        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(trajectory[:-1], dtrajectory)):
            dq, dp = func(q_base[i], p_base[i], m, t, **kwargs)
            q, p = solver(q_base[i], p_base[i], dq, dp, m, t, dt, **kwargs)
            q_traj[i], p_traj[i] = q, p

        return q_traj, p_traj


class EulerSolver(BaseSolver):
    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t, **kwargs)    # t1 --LAMMPSx2--> t2

        return q + dq * dt, p + dp * dt


class HyperEulerSolver(BaseSolver, HyperSolverMixin):
    trainable = True

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t, **kwargs)
        hq, hp = self.hypersolver(q, p, dq, dp, m, t, dt, **kwargs)

        q_next = q + dq * dt + hq * (dt ** 2)
        p_next = p + dp * dt + hp * (dt ** 2)

        return q_next, p_next

    def residual(self, func: Callable, q: torch.Tensor, q_next: torch.Tensor, p: torch.Tensor, p_next: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t, **kwargs)

        return (q_next - q - dq * dt) / (dt ** 2), (p_next - p - dp * dt) / (dt ** 2)

    def loss(self, func, q_base: torch.Tensor, p_base: torch.Tensor, m: torch.Tensor, trajectory, **kwargs):
        q_base, p_base, m, trajectory = q_base.detach(), p_base.detach(), m.detach(), trajectory.detach()

        q_res, p_res = self.residual_trajectory(func, q_base, p_base, m, trajectory, **kwargs)
        q_model, p_model = self.hypersolver_trajectory(self.hypersolver, func, q_base, p_base, m, trajectory, **kwargs)

        return torch.mean((q_res - q_model) ** 2) + torch.mean((p_res - p_model) ** 2)


class StormerVerletSolver(BaseSolver):
    def trajectory(self, func: Callable, q0: torch.Tensor, p0: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, **kwargs):
        q_traj = torch.zeros((trajectory.size(0), *q0.size()), dtype=q0.dtype, device=q0.device)

        dtrajectory = trajectory[1:] - trajectory[:-1]

        _, dp = func(q0, p0, m, 0, **kwargs)
        dt = dtrajectory[0]
        q1 = q0 + p0 / m * dt + (dp / (2 * m)) * (dt ** 2)

        q_traj[0], q_traj[1] = q0, q1

        for i, (t, dt) in enumerate(zip(trajectory[1:-1], dtrajectory[1:])):
            q0, q1 = self(func, q0, q1, m, t, dt, **kwargs)
            q_traj[i + 2] = q1

        return q_traj[:-1], (q_traj[1:] - q_traj[:-1]) * m

    def forward(self, func: Callable, q0: torch.Tensor, q1: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        _, dp = func(q1, (q1 - q0) * m, m, t, **kwargs)

        return q1, 2 * q1 - q0 + (dp / m) * (dt ** 2)


class HyperStormerVerletSolver(StormerVerletSolver):
    trainable = True

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, func: Callable, q0: torch.Tensor, q1: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q1, (q1 - q0) * m, m, t, **kwargs)
        hq, hp = self.hypersolver(q1, (q1 - q0) * m, dq, dp, m, t, dt, **kwargs)

        return q1, 2 * q1 - q0 + (dp / m) * (dt ** 2) + hq * (dt ** 3)


class VelocityVerletSolver(BaseSolver):
    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t, **kwargs)
        q_next = q + dq * dt + (dp / (2 * m)) * (dt ** 2)

        _, dp_next = func(q_next, p, m, t, **kwargs)
        p_next = p + ((dp + dp_next) / 2) * dt

        return q_next, p_next


class HyperVelocityVerletSolver(BaseSolver, HyperSolverMixin):
    trainable = True

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t, **kwargs)
        hq, hp = self.hypersolver(q, p, dq, dp, m, t, dt, include_p=False, **kwargs)
        q_next = q + dq * dt + (dp / (2 * m)) * (dt ** 2) + hq * (dt ** 2)

        dq_next, dp_next = func(q_next, p, m, t, **kwargs)
        hq, hp = self.hypersolver(q_next, p, dq_next, dp_next, m, t, dt, include_q=False, **kwargs)
        p_next = p + ((dp + dp_next) / 2) * dt + hp * (dt ** 2)

        return q_next, p_next

    def residual(self, func: Callable, q: torch.Tensor, q_next: torch.Tensor, p: torch.Tensor, p_next: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = func(q, p, m, t, **kwargs)
        q_prime = q + dq * dt + (dp / (2 * m)) * (dt ** 2)

        dq_next, dp_next = func(q_prime, p, m, t, **kwargs)
        p_prime = p + ((dp + dp_next) / 2) * dt

        return (q_next - q_prime) / (dt ** 2), (p_next - p_prime) / (dt ** 2)

    def hypersolver_trajectory(self, solver: Callable, func: Callable, q_base: torch.Tensor, p_base: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, **kwargs):
        q_traj = torch.zeros((trajectory.size(0) - 1, *q_base.shape[1:]), dtype=q_base.dtype, device=q_base.device)
        p_traj = torch.zeros((trajectory.size(0) - 1, *p_base.shape[1:]), dtype=p_base.dtype, device=p_base.device)

        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(trajectory[:-1], dtrajectory)):
            dq, dp = func(q_base[i], p_base[i], m, t, **kwargs)
            q_next = q_base[i] + dq * dt / m + (dp / (2 * m)) * (dt ** 2)

            dq_next, dp_next = func(q_next, p_base[i], m, t, **kwargs)
            hq, hp = solver(q_next, p_base[i], dq_next, dp_next, m, t, dt, **kwargs)

            q_traj[i], p_traj[i] = hq, hp

        return q_traj, p_traj

    def loss(self, func, q_base: torch.Tensor, p_base: torch.Tensor, m: torch.Tensor, trajectory, **kwargs):
        q_base, p_base, m, trajectory = q_base.detach(), p_base.detach(), m.detach(), trajectory.detach()

        q_res, p_res = self.residual_trajectory(func, q_base, p_base, m, trajectory, **kwargs)
        q_model, p_model = self.hypersolver_trajectory(self.hypersolver, func, q_base, p_base, m, trajectory, **kwargs)

        return torch.mean((q_res - q_model) ** 2) + torch.mean((p_res - p_model) ** 2)


class ThirdOrderRuthSolver(BaseSolver):
    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        c1 = 1
        c2 = -2 / 3
        c3 = 2 / 3
        d1 = -1 / 24
        d2 = 3 / 4
        d3 = 7 / 24

        q, p = q.detach(), p.detach()
        q = q + c1 * (p / m) * dt
        dq, dp = func(q, p, m, t, **kwargs)
        p = p + d1 * dp * dt

        q, p = q.detach(), p.detach()
        q = q + c2 * (p / m) * dt
        dq, dp = func(q, p, m, t, **kwargs)
        p = p + d2 * dp * dt

        q, p = q.detach(), p.detach()
        q = q + c3 * (p / m) * dt
        dq, dp = func(q, p, m, t, **kwargs)
        p = p + d3 * dp * dt

        return q, p


class FourthOrderRuthSolver(BaseSolver):
    def forward(self, func: Callable, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        two_power_one_third = 2 ** (1 / 3)
        c1 = 1 / (2 * (2 - two_power_one_third))
        c2 = (1 - two_power_one_third) / (2 * (2 - two_power_one_third))
        c3 = (1 - two_power_one_third) / (2 * (2 - two_power_one_third))
        c4 = 1 / (2 * (2 - two_power_one_third))
        d1 = 1 / (2 - two_power_one_third)
        d2 = two_power_one_third / (2 - two_power_one_third)
        d3 = 1 / (2 - two_power_one_third)
        d4 = 0

        q = q + c1 * (p / m) * dt
        dq, dp = func(q, p, m, t, **kwargs)
        p = p + d1 * dp * dt

        q = q + c2 * (p / m) * dt
        dq, dp = func(q, p, m, t, **kwargs)
        p = p + d2 * dp * dt

        q = q + c3 * (p / m) * dt
        dq, dp = func(q, p, m, t, **kwargs)
        p = p + d3 * dp * dt

        q = q + c4 * (p / m) * dt
        dq, dp = func(q, p, m, t, **kwargs)
        p = p + d4 * dp * dt

        return q, p
