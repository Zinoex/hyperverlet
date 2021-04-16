import numpy as np
from matplotlib import pyplot as plt, animation

from matplotlib.gridspec import GridSpec
from hyperverlet.energy import PendulumEnergy
from hyperverlet.plotting.energy import init_energy_plot, plot_energy, energy_animate_update
from hyperverlet.plotting.grid_spec import gs_3_2_3
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot
from hyperverlet.plotting.utils import save_animation
from hyperverlet.utils.misc import load_pickle, format_path


def pendulum_energy_plot(q, p, trajectory, m, l, g, plot_every=1):
    # Detatch and trim data
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    trajectory = trajectory.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()

    # Calculate energy of the system
    energy = PendulumEnergy()
    ke, pe, te = energy.all_energies(m, q, p, g=g, length=l)

    plot_energy(trajectory, te, ke, pe)


def animate_pendulum(config, show_gt=False, show_plot=True, cfg=0):
    plot_every = config["plotting"]["plot_every"]
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)
    save_plot = config["plotting"]["save_plot"]

    q = result_dict["q"][::plot_every, cfg]
    p = result_dict["p"][::plot_every, cfg]
    trajectory = result_dict["trajectory"][::plot_every, cfg]
    m = result_dict["mass"][cfg]
    l = result_dict["extra_args"]["length"][cfg]
    g = result_dict["extra_args"]["g"][cfg]

    # Ground Truth
    gt_q = result_dict["gt_q"][::plot_every, cfg]

    energy = PendulumEnergy()
    ke, pe, te = energy.all_energies(m, q, p, g=g, length=l)

    # Create grid spec
    fig = plt.figure(figsize=(20, 15))
    ax1, ax2, ax3 = gs_3_2_3(fig)

    # Initialize plots
    init_pendulum_plot(ax1, l)
    pe_plot, ke_plot, te_plot = init_energy_plot(ax2, trajectory, te, ke, pe)
    init_phasespace_plot(ax3, q, p)

    def animate(i):
        ax1.clear()
        update_pendulum(ax1, q, i, l, color="red")
        if show_gt:
            update_pendulum(ax1, gt_q, i, l, color="green", s=400, linewidth=2)

        energy_animate_update(ax2, pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te)
        update_phasespace_plot(ax3, q, p, i)

    anim = animation.FuncAnimation(fig, animate, frames=q.shape[0], save_count=q.shape[0])

    if show_plot:
        plt.show()

    if save_plot:
        save_animation(anim, config)


def update_pendulum(ax, q, i, l, color='red', s=500, linewidth=3):
    gt_x = l * np.sin(q[i])
    gt_y = - l * np.cos(q[i])
    ax.plot([0, gt_x[0]], [0, gt_y[0]], linewidth=linewidth, color=color)
    ax.scatter(gt_x, gt_y, color=color, marker='o', s=s, alpha=0.8)
    init_pendulum_plot(ax, l)


def init_pendulum_plot(ax, l):
    ax.set_title("Pendulum experiment")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([-l - 0.5, l + 0.5])
    ax.set_ylim([-l - 0.5, l + 0.5])
    ax.set_aspect('equal')
