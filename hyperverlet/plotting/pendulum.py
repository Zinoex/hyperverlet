import numpy as np
from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec
from hyperverlet.energy import pendulum
from hyperverlet.plotting.energy import init_energy_plot, update_energy_plot
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot


def pendulum_plot(q, p, time, m, g, l, plot_every=1):
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    time = time.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()

    pe = pendulum.calc_potential_energy(m, g, l, q)
    ke = pendulum.calc_kinetic_energy(m, l, p)
    te = pendulum.calc_total_energy(ke, pe)

    # Create grid spec
    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(2, 3)

    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    # PLOT - 2: Energy
    init_energy_plot(ax2, time, te, ke, pe)

    # PLOT - 3: Phase space setup
    init_phasespace_plot(ax3, q, p)

    for i in range(1, q.shape[0]):
        # PLOT - 1: Model
        ax1.clear()

        x = l * np.sin(q[i])
        y = - l * np.cos(q[i])

        ax1.plot([0, x[0]], [0, y[0]], linewidth=3)
        ax1.scatter(x, y, color='red', marker='o', s=500, alpha=0.8)
        ax1.set_xlabel('X', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Y', fontweight='bold', fontsize=14)
        ax1.set_xlim([-l - 0.5, l + 0.5])
        ax1.set_ylim([-l - 0.5, l + 0.5])
        ax1.set_aspect('equal')

        # PLOT - 2: Energy
        update_energy_plot(ax2, time, i, te, ke, pe)

        # PLOT - 3: Phase space
        update_phasespace_plot(ax3, q, p, i)

        plt.pause(1e-11)