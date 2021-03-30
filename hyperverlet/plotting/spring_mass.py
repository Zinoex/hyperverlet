import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from hyperverlet.energy import spring_mass
from hyperverlet.plotting.energy import init_energy_plot, update_energy_plot, plot_energy
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot
from hyperverlet.plotting.utils import plot_spring


def spring_mass_plot(result_dict, plot_every=1):
    q = result_dict["q"]
    p = result_dict["p"]
    trajectory = result_dict["trajectory"]
    m = result_dict["mass"]
    l = result_dict["extra_args"]["length"]
    k = result_dict["extra_args"]["k"]

    # Detatch and trim data
    q = q[::plot_every]
    p = p[::plot_every]
    trajectory = trajectory[::plot_every]
    k = k

    # Plotted bob circle radius
    r = 0.05
    wall_top = 0.5
    wall_bottom = -0.5

    # Calculate energy of the system
    ke = spring_mass.calc_kinetic_energy(m, p)
    pe = spring_mass.calc_potential_energy(k, q, l)
    te = spring_mass.calc_total_energy(ke, pe)

    # Create grid spec
    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    # PLOT - 2: Energy
    init_energy_plot(ax2, trajectory, te, ke, pe)

    # PLOT - 3: Phase space setup
    init_phasespace_plot(ax3, q, p)

    for i in range(1, q.shape[0]):
        # PLOT - 1: Model
        ax1.clear()

        ax1.set_xlim(-r, np.max(q) * 1.05 + r)
        ax1.set_ylim(wall_bottom * 1.05, wall_top * 1.05)
        ax1.set_aspect('equal')

        plot_spring(q[i], ax1)

        c0 = Circle((0, 0), r / 2, fc='k', zorder=10)
        c1 = Circle((q[i, 0], 0), r, fc='r', ec='r', zorder=10)
        ax1.add_patch(c0)
        ax1.add_patch(c1)
        # Add wall
        ax1.vlines(0, wall_bottom, wall_top, linestyles="solid", color='k', linewidth=7.0)

        # PLOT - 2: Energy
        update_energy_plot(ax2, trajectory, i, te, ke, pe)

        # PLOT - 3: Phase space
        update_phasespace_plot(ax3, q, p, i)

        plt.pause(1e-11)


def calc_theta(p1, p2):
    vec = p2 - p1

    return np.arctan2(vec[1], vec[0]) + np.pi / 2


def calc_dist_2d(p1, p2):
    return np.linalg.norm(p1 - p2)


def spring_mass_energy_plot(q, p, trajectory, m, k, l, plot_every=1):
    # Detatch and trim data
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    trajectory = trajectory.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()
    k = k.cpu().detach().numpy()

    # Calculate energy of the system
    ke = spring_mass.calc_kinetic_energy(m, p)
    pe = spring_mass.calc_potential_energy(k, q, l)
    te = spring_mass.calc_total_energy(ke, pe)

    plot_energy(trajectory, te, ke, pe)
