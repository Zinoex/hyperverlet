import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from hyperverlet.energy import pendulum, spring_mass


def lj_plot(time: np.array, pe: np.array, ke: np.array, te: np.array, dist: np.array, eps, sigma):
    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # PLOT - 2: Energy
    ax2.scatter(time[0], te[0], label=r'Total Energy$')
    ax2.scatter(time[0], ke[0], label=r'Kinetic Energy')
    ax2.scatter(time[0], pe[0], label=r'Potential Energy')
    ax2.legend()

    for i in range(1, len(time)):
        # PLOT - 1: Model
        ax1.clear()

        ax1.scatter(dist[i], 0, color='blue', marker='o',
                    s=500, alpha=0.8)
        ax1.set_xlim([0, 3])
        ax1.set_xlabel('X', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Y', fontweight='bold', fontsize=14)
        ax1.grid()

        # Potential energy curve
        r = np.linspace(0.8, 3, 100)
        U = np.array(eps * ((sigma / r) ** 12 - 2 * (sigma / r) ** 6))
        ax1.plot(r, U, 'r', label='LJ potential')
        ax1.legend()

        # PLOT - 2: Energy
        ax2.plot([time[i - 1], time[i]], [te[i - 1], te[i]], color='blue')
        ax2.plot([time[i - 1], time[i]], [ke[i - 1], ke[i]], color='orange')
        ax2.plot([time[i - 1], time[i]], [pe[i - 1], pe[i]], color='green')

        plt.pause(1E-11)


def mss_plot(time: np.array, q, p, gt_q, gt_p, label="Our Solver"):
    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(2, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax1.plot(time, gt_q, label='Analytical')
    ax1.plot(time, q, 'ro', alpha=0.5, label=label)
    ax1.set_ylabel('p(t)', fontweight='bold', fontsize=14)
    ax1.grid()
    ax1.legend()

    ax2.plot(time, gt_p)
    ax2.plot(time, p, 'ro', alpha=0.5)
    ax2.set_xlabel('time', fontweight='bold', fontsize=14)
    ax2.set_ylabel('q(t)', fontweight='bold', fontsize=14)
    ax2.grid()

    plt.show()


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
    ax2.scatter(time[0], te[0], label=r'$E_{sys}$')
    ax2.scatter(time[0], ke[0], label=r'KE')
    ax2.scatter(time[0], pe[0], label=r'PE')
    ax2.legend(loc="lower left")

    # PLOT - 3: Phase space
    ax3.set_title("Phase space")
    ax3.set_xlabel("q")
    ax3.set_ylabel("p")
    ax3.set_xlim(q.min() * 1.05, q.max() * 1.05)
    ax3.set_ylim(p.min() * 1.05, p.max() * 1.05)

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
        ax2.plot([time[i - 1], time[i]], [te[i - 1], te[i]], color='blue')
        ax2.plot([time[i - 1], time[i]], [ke[i - 1], ke[i]], color='orange')
        ax2.plot([time[i - 1], time[i]], [pe[i - 1], pe[i]], color='green')

        # PLOT - 3: Phase space
        ax3.plot(q[i], p[i], marker='x', color='black')

        plt.pause(1e-11)


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


def pendulum_plot(q, p, time, m, g, l, plot_every=1):
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    time = time.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()

    pe = pendulum.calc_potential_energy(m, g, l, q)
    ke = pendulum.calc_kinetic_energy(m, l, p)
    te = pendulum.calc_total_energy(ke, pe)

    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(2, 3)

    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    # PLOT - 3: Phase space
    ax3.set_title("Phase space")
    ax3.set_xlabel("q")
    ax3.set_ylabel("p")
    ax3.set_xlim(q.min() * 1.05, q.max() * 1.05)
    ax3.set_ylim(p.min() * 1.05, p.max() * 1.05)

    # PLOT - 2: Energy
    ax2.scatter(time[0], te[0], label=r'$E_{sys}$')
    ax2.scatter(time[0], ke[0], label=r'KE')
    ax2.scatter(time[0], pe[0], label=r'PE')
    ax2.legend(loc="lower left")

    for i in range(1, len(q)):
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
        ax2.plot([time[i - 1], time[i]], [te[i - 1], te[i]], color='blue')
        ax2.plot([time[i - 1], time[i]], [ke[i - 1], ke[i]],
                 color='orange')
        ax2.plot([time[i - 1], time[i]], [pe[i - 1], pe[i]],
                 color='green')

        # PLOT - 3: Phase space
        ax3.plot(q[i], p[i], marker='x', color='black')

        plt.pause(1e-11)


def plot_3d_pos(q, plot_every=1, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(q[::plot_every, :, 0].flatten(), q[::plot_every, :, 1].flatten(), q[::plot_every, :, 2].flatten(), marker='x')

    ax.set_title("Particle trajectories")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    if show:
        plt.show()


def plot_2d_pos(q, plot_every=1, show=True):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(q[::plot_every, :, 0].flatten(), q[::plot_every, :, 1].flatten(), marker='x')

    ax.set_title("Particle trajectories")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    if show:
        plt.show()


def plot_phasespace(q, p, show=True):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(p, q, marker='x')

    ax.set_title("Phase space")
    ax.set_xlabel("p")
    ax.set_ylabel("q")

    if show:
        plt.show()