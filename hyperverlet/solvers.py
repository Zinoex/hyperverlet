from typing import Callable

import torch
from torch import nn
from tqdm import tqdm

from hyperverlet.experiments import Experiment


class BaseSolver(nn.Module):
    trainable = False
    residual_training = False

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        raise NotImplemented()

    def trajectory(self, experiment: Experiment, q0: torch.Tensor, p0: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = torch.zeros((trajectory.size(0), *q0.size()), dtype=q0.dtype, device=q0.device)
        p_traj = torch.zeros((trajectory.size(0), *p0.size()), dtype=p0.dtype, device=p0.device)

        q_traj[0], p_traj[0] = q0, p0
        q, p = experiment.shift(q0, **kwargs), p0

        trajectory = self.view_like_q(trajectory, q_traj)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = self(experiment, q, p, m, t, dt, **kwargs)
            q_traj[i + 1], p_traj[i + 1] = q, p

        return q_traj, p_traj

    def view_like_q(self, trajectory, q):
        traj_size = list(trajectory.size())
        remaining_size = [1] * (len(q.size()) - len(traj_size))

        return trajectory.view(*traj_size, *remaining_size)


class Euler(BaseSolver):
    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = experiment(q, p, m, t, **kwargs)

        return experiment.shift(q + dq * dt, **kwargs), p + dp * dt


class ResidualMixin:
    def base_trajectory(self, experiment: Experiment, gt_q: torch.Tensor, gt_p: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, dtrajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = torch.zeros_like(gt_q)
        p_traj = torch.zeros_like(gt_p)

        q_traj[0], p_traj[0] = gt_q[0], gt_p[0]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = self.base(experiment, gt_q[i], gt_p[i], m, t, dt, **kwargs)
            q_traj[i + 1], p_traj[i + 1] = q, p

        return q_traj, p_traj

    def hyper_trajectory(self, experiment: Experiment, gt_q: torch.Tensor, gt_p: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = torch.zeros_like(gt_q)
        p_traj = torch.zeros_like(gt_p)

        gt_q = experiment.shift(gt_q, **kwargs)

        trajectory = self.view_like_q(trajectory, gt_q)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            q, p = gt_q[i], gt_p[i]
            dq1, dp1 = experiment(q, p, m, t, **kwargs)

            q, p = self.base(experiment, q, p, m, t, dt, **kwargs)

            dq2, dp2 = experiment(q, p, m, t, **kwargs)
            hq, hp = self.hypersolver(dq1, dq2, dp1, dp2, m, t, dt, **kwargs)

            q_traj[i], p_traj[i] = hq, hp

        return q_traj[:-1], p_traj[:-1]

    def get_residuals(self, experiment, gt_q, gt_p, m, trajectory, **kwargs):
        trajectory = self.view_like_q(trajectory, gt_q)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        gt_q = experiment.shift(gt_q, **kwargs)

        q_pred, p_pred = self.base_trajectory(experiment, gt_q, gt_p, m, trajectory, dtrajectory, **kwargs)
        res_q = (gt_q[1:] - q_pred[1:]) / dtrajectory ** (self.p_order + 1)
        res_p = (gt_p[1:] - p_pred[1:]) / dtrajectory ** (self.p_order + 1)
        return res_q, res_p


class HyperEuler(BaseSolver, ResidualMixin):
    trainable = True
    residual_training = True
    p_order = 1

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = experiment(q, p, m, t, **kwargs)
        hq, hp = self.hypersolver(q, p, dq, dp, m, t, dt, **kwargs)

        q_next = experiment.shift(q + dq * dt + hq * (dt ** 2), **kwargs)
        p_next = p + dp * dt + hp * (dt ** 2)

        return q_next, p_next

    def base(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = experiment(q, p, m, t, **kwargs)

        q_next = experiment.shift(q + dq * dt, **kwargs)
        p_next = p + dp * dt

        return q_next, p_next


class Heun(BaseSolver):
    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = experiment(q, p, m, t, **kwargs)
        q_hat, p_hat = experiment.shift(q + dq * dt, **kwargs), p + dp * dt
        dq_hat, dp_hat = experiment(q_hat, p_hat, m, t + dt, **kwargs)

        return experiment.shift(q + dt / 2 * (dq + dq_hat), **kwargs), p + dt / 2 * (dp + dp_hat)


class HyperHeun(BaseSolver):
    trainable = True

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq, dp = experiment(q, p, m, t, **kwargs)
        hq, hp = self.hypersolver(q, p, dq, dp, m, t, dt, **kwargs)
        q_hat, p_hat = experiment.shift(q + dq * dt + (dt ** 2) * hq, **kwargs), p + dp * dt + (dt ** 2) * hp
        dq_hat, dp_hat = experiment(q_hat, p_hat, m, t + dt, **kwargs)
        hq, hp = self.hypersolver(q_hat, p_hat, dq_hat, dp_hat, m, t + dt, dt, **kwargs)

        return experiment.shift(q + dt / 2 * (dq + dq_hat) + (dt ** 3) * hq, **kwargs), p + dt / 2 * (dp + dp_hat) + (dt ** 3) * hp


class RungeKutta4(BaseSolver):
    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq1, dp1 = experiment(q, p, m, t, **kwargs)
        q1, p1, t1 = experiment.shift(q + dq1 * dt / 2, **kwargs), p + dp1 * dt / 2, t + dt / 2
        dq2, dp2 = experiment(q1, p1, m, t1, **kwargs)
        q2, p2, t2 = experiment.shift(q + dq2 * dt / 2, **kwargs), p + dp2 * dt / 2, t + dt / 2
        dq3, dp3 = experiment(q2, p2, m, t2, **kwargs)
        q3, p3, t3 = experiment.shift(q + dq3 * dt, **kwargs), p + dp3 * dt, t + dt
        dq4, dp4 = experiment(q3, p3, m, t3, **kwargs)

        return experiment.shift(q + (dq1 + 2 * dq2 + 2 * dq3 + dq4) / 6 * dt, **kwargs), p + (dp1 + 2 * dp2 + 2 * dp3 + dp4) / 6 * dt


class SymplecticEuler(BaseSolver):
    def __init__(self):
        super().__init__()

        self.q_first = True

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        foward_func = self.get_forward_func()

        return foward_func(experiment, q, p, m, t, dt, **kwargs)

    def get_forward_func(self):
        return self.forward_q if self.q_first else self.forward_p

    def forward_q(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq = experiment.dq(p, m, t, **kwargs)
        q = experiment.shift(q + dq * dt, **kwargs)

        dp = experiment.dp(q, m, t, **kwargs)
        p = p + dp * dt

        return q, p

    def forward_p(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dp = experiment.dp(q, m, t, **kwargs)
        p = p + dp * dt

        dq = experiment.dq(p, m, t, **kwargs)
        q = experiment.shift(q + dq * dt, **kwargs)

        return q, p


class AlternatingSymplecticEuler(SymplecticEuler):
    def get_forward_func(self):
        foward_func = self.forward_q if self.q_first else self.forward_p
        self.q_first = not self.q_first

        return foward_func


class HyperSymplecticEuler(BaseSolver):
    trainable = True

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver
        self.q_first = True

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        foward_func = self.get_forward_func()

        return foward_func(experiment, q, p, m, t, dt, **kwargs)

    def get_forward_func(self):
        foward_func = self.forward_q if self.q_first else self.forward_p
        self.q_first = not self.q_first

        return foward_func

    def forward_q(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq1, dp1 = experiment(q, p, m, t, **kwargs)
        dq = experiment.dq(p, m, t, **kwargs)
        q = experiment.shift(q + dq * dt, **kwargs)

        dq2, dp2 = experiment(q, p, m, t, **kwargs)
        hq = self.hypersolver.hq(dq1, dp1, dq2, dp2, m, t, dt, **kwargs)
        q = experiment.shift(q + hq * dt, **kwargs)

        dq1, dp1 = experiment(q, p, m, t, **kwargs)
        dp = experiment.dp(q, m, t, **kwargs)
        p = p + dp * dt

        dq2, dp2 = experiment(q, p, m, t, **kwargs)
        hp = self.hypersolver.hp(dq1, dp1, dq2, dp2, m, t, dt, **kwargs)
        p = p + hp * dt
        return q, p

    def forward_p(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq1, dp1 = experiment(q, p, m, t, **kwargs)
        dp = experiment.dp(q, m, t, **kwargs)
        p = p + dp * dt

        dq2, dp2 = experiment(q, p, m, t, **kwargs)
        hp = self.hypersolver.hp(dq1, dp1, dq2, dp2, m, t, dt, **kwargs)
        p = p + hp * dt

        dq1, dp1 = experiment(q, p, m, t, **kwargs)
        dq = experiment.dq(p, m, t, **kwargs)
        q = q + dq * dt

        dq2, dp2 = experiment(q, p, m, t, **kwargs)
        hq = self.hypersolver.hq(dq1, dp1, dq2, dp2, m, t, dt, **kwargs)
        q = experiment.shift(q + hq * dt, **kwargs)
        return q, p


class HyperVelocityVerlet(BaseSolver, ResidualMixin):
    trainable = True
    residual_trainable = True
    p_order = 2

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dq1, dp1 = experiment(q, p, m, t, **kwargs)

        q, p = self.base(experiment, q, p, m, t, dt, **kwargs)

        dq2, dp2 = experiment(q, p, m, t, **kwargs)
        hq, hp = self.hypersolver(dq1, dq2, dp1, dp2, m, t, dt, **kwargs)
        q, p = experiment.shift(q + hq * dt ** (self.p_order + 1), **kwargs), p + hp * dt ** (self.p_order + 1)

        return q, p

    def base(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        one_half = 1 / 2

        dp = experiment.dp(q, m, t, **kwargs)
        p = p + one_half * dp * dt

        dq = experiment.dq(p, m, t, **kwargs)
        q = experiment.shift(q + dq * dt, **kwargs)

        dp = experiment.dp(q, m, t, **kwargs)
        p = p + one_half * dp * dt

        return q, p


class SymplecticSolver(BaseSolver):
    def __init__(self, c, d):
        super().__init__()

        assert len(c) == len(d)

        self.c = c
        self.d = d

    def forward(self, experiment: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        for c, d in zip(self.c, self.d):
            if c != 0:
                dq = experiment.dq(p, m, t, **kwargs)
                q = experiment.shift(q + c * dq * dt, **kwargs)

            if d != 0:
                dp = experiment.dp(q, m, t, **kwargs)
                p = p + d * dp * dt

        return q, p


class VelocityVerlet(SymplecticSolver):
    def __init__(self):
        one_half = 1 / 2

        super().__init__(
            c=[0, 1],
            d=[one_half, one_half]
        )


class ThirdOrderRuth(SymplecticSolver):
    def __init__(self):
        c1 = 1
        c2 = -2 / 3
        c3 = 2 / 3
        d1 = -1 / 24
        d2 = 3 / 4
        d3 = 7 / 24

        super().__init__(
            c=[c1, c2, c3],
            d=[d1, d2, d3]
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