import numpy as np
from matplotlib import pyplot as plt, animation

from matplotlib.gridspec import GridSpec
from hyperverlet.energy import pendulum
from hyperverlet.plotting.energy import init_energy_plot, update_energy_plot, plot_energy, energy_animate_update
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot


def pendulum_plot(result_dict, plot_every=1, show_gt=False):
    q = result_dict["q"][::plot_every]
    p = result_dict["p"][::plot_every]
    trajectory = result_dict["trajectory"][::plot_every]
    m = result_dict["mass"]
    l = result_dict["extra_args"]["length"]
    g = result_dict["extra_args"]["g"]

    # Ground Truth
    gt_q = np.squeeze(result_dict["gt_q"][::plot_every], axis=1)

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
    init_energy_plot(ax2, trajectory, te, ke, pe)

    # PLOT - 3: Phase space setup
    init_phasespace_plot(ax3, q, p)

    for i in range(1, q.shape[0]):
        # PLOT - 1: Model
        ax1.clear()

        x = l * np.sin(q[i])
        y = - l * np.cos(q[i])

        ax1.plot([0, x[0]], [0, y[0]], linewidth=3, color='red')
        ax1.scatter(x, y, color='red', marker='o', s=500, alpha=0.8)
        init_pendulum_plot(ax1, l)

        if show_gt:
            gt_x = l * np.sin(gt_q[i])
            gt_y = - l * np.cos(gt_q[i])
            ax1.plot([0, gt_x[0]], [0, gt_y[0]], linewidth=2, color='green')
            ax1.scatter(gt_x, gt_y, color='green', marker='o', s=400, alpha=0.8)

        # PLOT - 2: Energy
        update_energy_plot(ax2, trajectory, i, te, ke, pe)

        # PLOT - 3: Phase space
        update_phasespace_plot(ax3, q, p, i)

        plt.pause(1e-11)


def spring_mass_energy_plot(q, p, trajectory, m, k, l, g, plot_every=1):
    # Detatch and trim data
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    trajectory = trajectory.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()
    k = k.cpu().detach().numpy()

    # Calculate energy of the system
    pe = pendulum.calc_potential_energy(m, g, l, q)
    ke = pendulum.calc_kinetic_energy(m, l, p)
    te = pendulum.calc_total_energy(ke, pe)

    plot_energy(trajectory, te, ke, pe)


def animate_pendulum(result_dict, plot_every=1, show_gt=False):
    q = result_dict["q"][::plot_every]
    p = result_dict["p"][::plot_every]
    trajectory = result_dict["trajectory"][::plot_every]
    interval = trajectory[1] - trajectory[0]
    m = result_dict["mass"]
    l = result_dict["extra_args"]["length"]
    g = result_dict["extra_args"]["g"]

    # Ground Truth
    gt_q = np.squeeze(result_dict["gt_q"][::plot_every], axis=1)

    pe = pendulum.calc_potential_energy(m, g, l, q)
    ke = pendulum.calc_kinetic_energy(m, l, p)
    te = pendulum.calc_total_energy(ke, pe)

    # Create grid spec
    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(2, 3)

    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    # Initialize plots
    init_pendulum_plot(ax1, l)
    pe_plot, ke_plot, te_plot = init_energy_plot(ax2, trajectory, te, ke, pe)
    init_phasespace_plot(ax3, q, p)

    def animate(i):
        ax1.clear()
        update_pendulum(ax1, q, i, l, color="red")
        if show_gt:
            update_pendulum(ax1, gt_q, i, l, color="green", s=400, linewidth=2)

        energy_animate_update(pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te, ax2)
        update_phasespace_plot(ax3, q, p, i)

    anim = animation.FuncAnimation(fig, animate, frames=q.shape[0])

    plt.show()


def update_pendulum(ax, q, i, l, color='red', s=500, linewidth=3):
    gt_x = l * np.sin(q[i])
    gt_y = - l * np.cos(q[i])
    ax.plot([0, gt_x[0]], [0, gt_y[0]], linewidth=linewidth, color=color)
    ax.scatter(gt_x, gt_y, color=color, marker='o', s=s, alpha=0.8)
    init_pendulum_plot(ax, l)


def init_pendulum_plot(ax, l):
    ax.set_title("Pendulum experiment")
    ax.set_xlabel('X', fontweight='bold', fontsize=14)
    ax.set_ylabel('Y', fontweight='bold', fontsize=14)
    ax.set_xlim([-l - 0.5, l + 0.5])
    ax.set_ylim([-l - 0.5, l + 0.5])
    ax.set_aspect('equal')