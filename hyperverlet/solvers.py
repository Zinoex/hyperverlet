from typing import Callable

import torch
from torch import nn
from tqdm import tqdm

from hyperverlet.experiments import Experiment


class BaseSolver(nn.Module):
    trainable = False

    def forward(self, func: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        raise NotImplemented()

    def trajectory(self, func: Experiment, q0: torch.Tensor, p0: torch.Tensor, m: torch.Tensor, trajectory: torch.Tensor, disable_print=False, **kwargs):
        q_traj = torch.zeros((trajectory.size(0), *q0.size()), dtype=q0.dtype, device=q0.device)
        p_traj = torch.zeros((trajectory.size(0), *p0.size()), dtype=p0.dtype, device=p0.device)

        q_traj[0], p_traj[0] = q0, p0
        q, p = q0, p0

        trajectory = trajectory.unsqueeze(-1)
        dtrajectory = trajectory[1:] - trajectory[:-1]

        for i, (t, dt) in enumerate(zip(tqdm(trajectory[:-1], disable=disable_print), dtrajectory)):
            # q, p = q.detach(), p.detach()
            q, p = self(func, q, p, m, t, dt, **kwargs)
            q_traj[i + 1], p_traj[i + 1] = q, p

        return q_traj, p_traj


class Euler(BaseSolver):
    def forward(self, func: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dt = dt.view(-1, *[1 for _ in range(len(q.size()) - 1)])

        dq, dp = func(q, p, m, t, **kwargs)

        return q + dq * dt, p + dp * dt


class HyperEuler(BaseSolver):
    trainable = True

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, func: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dt = dt.view(-1, *[1 for _ in range(len(q.size()) - 1)])

        dq, dp = func(q, p, m, t, **kwargs)
        hq, hp = self.hypersolver(q, p, dq, dp, m, t, dt, **kwargs)

        q_next = q + dq * dt + hq * (dt ** 2)
        p_next = p + dp * dt + hp * (dt ** 2)

        return q_next, p_next


class Heun(BaseSolver):
    def forward(self, func: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dt = dt.view(-1, *[1 for _ in range(len(q.size()) - 1)])

        dq, dp = func(q, p, m, t, **kwargs)
        q_hat, p_hat = q + dq * dt, p + dp * dt
        dq_hat, dp_hat = func(q_hat, p_hat, m, t + dt, **kwargs)

        return q + dt / 2 * (dq + dq_hat), p + dt / 2 * (dp + dp_hat)


class HyperHeun(BaseSolver):
    trainable = True

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, func: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dt = dt.view(-1, *[1 for _ in range(len(q.size()) - 1)])

        dq, dp = func(q, p, m, t, **kwargs)
        hq, hp = self.hypersolver(q, p, dq, dp, m, t, dt, **kwargs)
        q_hat, p_hat = q + dq * dt + (dt ** 2) * hq, p + dp * dt + (dt ** 2) * hp
        dq_hat, dp_hat = func(q_hat, p_hat, m, t + dt, **kwargs)
        hq, hp = self.hypersolver(q_hat, p_hat, dq_hat, dp_hat, m, t + dt, dt, **kwargs)

        return q + dt / 2 * (dq + dq_hat) + (dt ** 3) * hq, p + dt / 2 * (dp + dp_hat) + (dt ** 3) * hp


class RungeKutta4(BaseSolver):
    def forward(self, func: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dt = dt.view(-1, *[1 for _ in range(len(q.size()) - 1)])

        dq1, dp1 = func(q, p, m, t, **kwargs)
        q1, p1, t1 = q + dq1 * dt / 2, p + dp1 * dt / 2, t + dt / 2
        dq2, dp2 = func(q1, p1, m, t1, **kwargs)
        q2, p2, t2 = q + dq2 * dt / 2, p + dp2 * dt / 2, t + dt / 2
        dq3, dp3 = func(q2, p2, m, t2, **kwargs)
        q3, p3, t3 = q + dq3 * dt, p + dp3 * dt, t + dt
        dq4, dp4 = func(q3, p3, m, t3, **kwargs)

        return q + (dq1 + 2 * dq2 + 2 * dq3 + dq4) / 6 * dt, p + (dp1 + 2 * dp2 + 2 * dp3 + dp4) / 6 * dt


class HyperVelocityVerlet(BaseSolver):
    trainable = True

    def __init__(self, hypersolver):
        super().__init__()

        self.hypersolver = hypersolver

    def forward(self, func: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dt = dt.view(-1, *[1 for _ in range(len(q.size()) - 1)])
        one_half = 1 / 2

        dp = func.dp(q, m, t, **kwargs)
        p = p + one_half * dp * dt

        dq = func.dq(p, m, t, **kwargs)
        q = q + dq * dt

        dp = func.dp(q, m, t, **kwargs)
        p = p + one_half * dp * dt

        hq, hp = self.hypersolver(q, p, dq, dp, m, t, dt, **kwargs)
        return q + hq * dt ** 2, p + hp * dt ** 2


class SymplecticSolver(BaseSolver):
    def __init__(self, c, d):
        super().__init__()

        assert len(c) == len(d)

        self.c = c
        self.d = d

    def forward(self, func: Experiment, q: torch.Tensor, p: torch.Tensor, m: torch.Tensor, t, dt, **kwargs):
        dt = dt.view(-1, *[1 for _ in range(len(q.size()) - 1)])

        for c, d in zip(self.c, self.d):
            if c != 0:
                dq = func.dq(p, m, t, **kwargs)
                q = q + c * dq * dt

            if d != 0:
                dp = func.dp(q, m, t, **kwargs)
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
