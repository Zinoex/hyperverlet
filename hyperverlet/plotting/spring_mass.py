import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from hyperverlet.energy import spring_mass
from hyperverlet.plotting.energy import init_energy_plot, update_energy_plot
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot


def _plot_spring(l, ax):
    """Plot the spring from (0,0) to (x,y) as the projection of a helix."""
    theta = np.pi / 2

    # Spring turn radius, number of turns
    rs, ns = 0.05, 25
    # Number of data points for the helix
    Ns = 1000
    # We don't draw coils all the way to the end of the pendulum:
    # pad a bit from the anchor and from the bob by these number of points
    ipad1, ipad2 = 100, 150
    w = np.linspace(0, l, Ns).flatten()
    # Set up the helix along the x-axis ...
    xp = np.zeros(Ns)
    xp[ipad1:-ipad2] = rs * np.sin(2 * np.pi * ns * w[ipad1:-ipad2] / l)
    # ... then rotate it to align with  the pendulum and plot.
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    xs, ys = - R @ np.vstack((xp, w))
    ax.plot(xs, ys, c='k', lw=2)


def spring_mass_plot(q, p, time, m, k, l, plot_every=1):
    # Detatch and trim data
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    time = time.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()
    k = k.cpu().detach().numpy()

    # Plotted bob circle radius
    r = 0.05
    wall_top = 0.5
    wall_bottom = -0.5

    # Calculate energy of the system
    pe = spring_mass.calc_potential_energy(m, p)
    ke = spring_mass.calc_kinetic_energy(k, q, l)
    te = spring_mass.calc_total_energy(ke, pe)

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

        _plot_spring(q[i], ax1)

        c0 = Circle((0, 0), r / 2, fc='k', zorder=10)
        c1 = Circle((q[i, 0], 0), r, fc='r', ec='r', zorder=10)
        ax1.add_patch(c0)
        ax1.add_patch(c1)
        # Add wall
        ax1.vlines(0, wall_bottom, wall_top, linestyles="solid", color='k', linewidth=7.0)

        ax1.set_xlim(-r, np.max(q) * 1.05 + r)
        ax1.set_ylim(wall_bottom * 1.05, wall_top * 1.05)
        ax1.set_aspect('equal')

        # PLOT - 2: Energy
        update_energy_plot(ax2, time, i, te, ke, pe)

        # PLOT - 3: Phase space
        update_phasespace_plot(ax3, q, p, i)

        plt.pause(1e-11)