from typing import Callable

import torch
from torch import nn
from tqdm import tqdm

from hyperverlet.experiments import Experiment


class BaseSolver(nn.Module):
    trainable = False
    residual_trainable = False

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        raise NotImplemented()

    def trajectory(self, experiment: Experiment, q0: torch.Tensor, p0: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = [q0]
        p_traj = [p0]

        q, p = experiment.shift(q0, **kwargs), p0

        trajectory = self.view_like_q(trajectory, q0.dim() + 1)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = self(experiment, q, p, m, t, dt, **kwargs)
            q_traj.append(q)
            p_traj.append(p)

        return torch.stack(q_traj, 0), torch.stack(p_traj, 0)

    def view_like_q(self, trajectory, q):
        traj_size = list(trajectory.size())
        q_size = q.dim() if torch.is_tensor(q) else q
        remaining_size = [1] * (q_size - len(traj_size))

        return trajectory.view(*traj_size, *remaining_size)


class Euler(BaseSolver):
    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = experiment(q, p, m, t, **kwargs)

        return experiment.shift(q + dq * dt, **kwargs), p + dp * dt


class ResidualMixin:
    def hyper_forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq1, dp1 = experiment(q, p, m, t, **kwargs)

        q, p = self.base(experiment, q, p, m, t, dt, **kwargs)

        dq2, dp2 = experiment(q, p, m, t, **kwargs)
        hq, hp = self.hypersolver(dq1, dq2, dp1, dp2, m, t, dt, **kwargs)
        q, p = experiment.shift(q + hq * dt ** (self.q_order + 1), **kwargs), p + hp * dt ** (self.p_order + 1)

        return q, p

    def base_trajectory(self, experiment: Experiment, gt_q: torch.Tensor, gt_p: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, dtrajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = [gt_q[0]]
        p_traj = [gt_p[0]]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = self.base(experiment, gt_q[i], gt_p[i], m, t, dt, **kwargs)
            q_traj.append(q)
            p_traj.append(p)

        return torch.stack(q_traj, 0), torch.stack(p_traj, 0)

    def hyper_trajectory(self, experiment: Experiment, gt_q: torch.Tensor, gt_p: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = []
        p_traj = []

        gt_q = experiment.shift(gt_q, **kwargs)

        trajectory = self.view_like_q(trajectory, gt_q)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = gt_q[i], gt_p[i]
            dq1, dp1 = experiment(q, p, m, t, **kwargs)

            q, p = self.base(experiment, q, p, m, t, dt, **kwargs)

            dq2, dp2 = experiment(q, p, m, t, **kwargs)
            hq, hp = self.hypersolver(dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

            q_traj.append(hq)
            p_traj.append(hp)

        return torch.stack(q_traj, 0), torch.stack(p_traj, 0)

    def get_residuals(self, experiment, gt_q, gt_p, m, trajectory, **kwargs):
        trajectory = self.view_like_q(trajectory, gt_q)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        gt_q = experiment.shift(gt_q, **kwargs)

        q_pred, p_pred = self.base_trajectory(experiment, gt_q, gt_p, m, trajectory, dtrajectory, **kwargs)
        res_q = (gt_q[1:] - q_pred[1:]) / dtrajectory ** (self.q_order + 1)
        res_p = (gt_p[1:] - p_pred[1:]) / dtrajectory ** (self.p_order + 1)
        return res_q, res_p


class HyperEuler(BaseSolver, ResidualMixin):
    trainable = True
    q_order = 1
    p_order = 1

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        return self.hyper_forward(experiment, q, p, m, t, dt, **kwargs)

    def base(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = experiment(q, p, m, t, **kwargs)

        q_next = experiment.shift(q + dq * dt, **kwargs)
        p_next = p + dp * dt

        return q_next, p_next


class HyperVelocityVerlet(BaseSolver, ResidualMixin):
    trainable = True
    q_order = 2
    p_order = 2

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        return self.hyper_forward(experiment, q, p, m, t, dt, **kwargs)

    def base(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        one_half = 1 / 2

        dp = experiment.dp(q, p, m, t, **kwargs)
        p = p + one_half * dp * dt

        dq = experiment.dq(q, p, m, t, **kwargs)
        q = experiment.shift(q + dq * dt, **kwargs)

        dp = experiment.dp(q, p, m, t, **kwargs)
        p = p + one_half * dp * dt

        return q, p


class SymplecticHyperVelocityVerlet(BaseSolver, ResidualMixin):
    trainable = True
    q_order = 2
    p_order = 2

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        q, p = self.base(experiment, q, p, m, t, dt, **kwargs)
        q, p = self.hypersolver(q, p, m, t, dt, **kwargs)

        return q, p

    def base(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        one_half = 1 / 2

        dp = experiment.dp(q, p, m, t, **kwargs)
        p = p + one_half * dp * dt

        dq = experiment.dq(q, p, m, t, **kwargs)
        q = experiment.shift(q + dq * dt, **kwargs)

        dp = experiment.dp(q, p, m, t, **kwargs)
        p = p + one_half * dp * dt

        return q, p

    def hyper_trajectory(self, experiment: Experiment, gt_q: torch.Tensor, gt_p: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = []
        p_traj = []

        gt_q = experiment.shift(gt_q, **kwargs)

        trajectory = self.view_like_q(trajectory, gt_q)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = gt_q[i], gt_p[i]

            q_base, p_base = self.base(experiment, q, p, m, t, dt, **kwargs)
            q, p = self.hypersolver(q_base, p_base, m, t, dt, **kwargs)

            q_traj.append(q - q_base)
            p_traj.append(p - p_base)

        return torch.stack(q_traj, 0), torch.stack(p_traj, 0)

    def get_residuals(self, experiment, gt_q, gt_p, m, trajectory, **kwargs):
        trajectory = self.view_like_q(trajectory, gt_q)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        gt_q = experiment.shift(gt_q, **kwargs)

        q_pred, p_pred = self.base_trajectory(experiment, gt_q, gt_p, m, trajectory, dtrajectory, **kwargs)
        res_q = gt_q[1:] - q_pred[1:]
        res_p = gt_p[1:] - p_pred[1:]
        return res_q, res_p


class SympNet(BaseSolver, ResidualMixin):
    trainable = True
    q_order = 2
    p_order = 2

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        q, p = self.hypersolver(q, p, m, t, dt, **kwargs)

        return q, p

    def hyper_trajectory(self, experiment: Experiment, gt_q: torch.Tensor, gt_p: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = []
        p_traj = []

        gt_q = experiment.shift(gt_q, **kwargs)

        trajectory = self.view_like_q(trajectory, gt_q)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = gt_q[i], gt_p[i]
            q_pred, p_pred = self.hypersolver(q, p, m, t, dt, **kwargs)

            q_traj.append(q_pred - q)
            p_traj.append(p_pred - p)

        return torch.stack(q_traj, 0), torch.stack(p_traj, 0)

    def get_residuals(self, experiment, gt_q, gt_p, m, trajectory, **kwargs):
        gt_q = experiment.shift(gt_q, **kwargs)

        res_q = gt_q[1:] - gt_q[:-1]
        res_p = gt_p[1:] - gt_p[:-1]
        return res_q, res_p


class SymplecticSolver(BaseSolver):
    def __init__(self, c, d):
        super().__init__()

        assert len(c) == len(d)

        self.c = c
        self.d = d

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        return self.base(experiment, q, p, m, t, dt, **kwargs)

    def base(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        for c, d in zip(self.c, self.d):
            if c != 0:
                dq = experiment.dq(q, p, m, t, **kwargs)
                q = experiment.shift(q + c * dq * dt, **kwargs)

            if d != 0:
                dp = experiment.dp(q, p, m, t, **kwargs)
                p = p + d * dp * dt

        return q, p


class VelocityVerlet(SymplecticSolver):
    def __init__(self):
        one_half = 1 / 2

        super().__init__(
            c=[0, 1],
            d=[one_half, one_half]
        )


class FourthOrderRuth(SymplecticSolver):
    def __init__(self):
        two_power_one_third = 2 ** (1 / 3)
        two_minus_two_power_one_third = 2 - two_power_one_third

        c1 = 1 / (2 * two_minus_two_power_one_third)
        c2 = (1 - two_power_one_third) / (2 * two_minus_two_power_one_third)
        c3 = c2
        c4 = c1
        d1 = 1 / two_minus_two_power_one_third
        d2 = -two_power_one_third / two_minus_two_power_one_third
        d3 = d1
        d4 = 0

        super().__init__(
            c=[c1, c2, c3, c4],
            d=[d1, d2, d3, d4]
        )
