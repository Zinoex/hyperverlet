import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from energy.pendul_energy import calc_potential_energy, calc_kinetic_energy, calc_total_energy


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


def pendulum_plot(time, m, g, l, q, p, plot_every=1):
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    time = time.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()

    pe = calc_potential_energy(m, g, l, q)
    ke = calc_kinetic_energy(m, l, p)
    te = calc_total_energy(ke, pe)

    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # PLOT - 2: Energy
    ax2.scatter(time[0], te[0], label=r'$E_{sys}$')
    ax2.scatter(time[0], ke[0], label=r'KE')
    ax2.scatter(time[0], pe[0], label=r'PE')
    ax2.legend()

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