import numpy as np
from matplotlib import pyplot as plt, animation

from matplotlib.gridspec import GridSpec
from hyperverlet.energy import PendulumEnergy
from hyperverlet.plotting.energy import init_energy_plot, plot_energy, energy_animate_update
from hyperverlet.plotting.grid_spec import gs_3_2_3
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot
from hyperverlet.plotting.utils import save_animation
from hyperverlet.utils.misc import load_pickle, format_path


def pendulum_energy_plot(q, p, trajectory, m, length, g, plot_every=1):
    # Detatch and trim data
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    trajectory = trajectory.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    length = length.cpu().detach().numpy()

    # Calculate energy of the system
    energy = PendulumEnergy()
    ke, pe, te = energy.all_energies(m, q, p, g=g, length=length)

    plot_energy(trajectory, te, ke, pe)


def animate_pendulum(config, show_gt=False, show_plot=True, cfg=1):
    plot_every = config["plotting"]["plot_every"]
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)
    save_plot = config["plotting"]["save_plot"]

    q = result_dict["q"][::plot_every, cfg]
    p = result_dict["p"][::plot_every, cfg]
    trajectory = result_dict["trajectory"][::plot_every, cfg]
    mass = result_dict["mass"][cfg]
    length = result_dict["extra_args"]["length"][cfg]
    g = result_dict["extra_args"]["g"][cfg]

    x = length * np.sin(q)
    y = -length * np.cos(q)

    # Ground Truth
    gt_q = result_dict["gt_q"][::plot_every, cfg]
    gt_x = length * np.sin(gt_q)
    gt_y = -length * np.cos(gt_q)

    # Energy
    energy = PendulumEnergy()
    ke, pe, te = energy.all_energies(mass, q, p, g=g, length=length)

    # Create grid spec
    fig = plt.figure(figsize=(20, 15))
    ax_pendulum, ax_energy, ax_phase_space = gs_3_2_3(fig)

    # Initialize plots
    init_pendulum_plot(ax_pendulum, x, length)
    pe_plot, ke_plot, te_plot = init_energy_plot(ax_energy, trajectory, te, ke, pe)
    ps_plot = init_phasespace_plot(ax_phase_space, q, p)

    line, scatter = init_pendulum(ax_pendulum, x, y, color="red", zorder=1)
    if show_gt:
        gt_line, gt_scatter = init_pendulum(ax_pendulum, gt_x, gt_y, color="green", s=300, linewidth=2, zorder=2)

    def animate(i):
        update_pendulum(line, scatter, x, y, i)
        if show_gt:
            update_pendulum(gt_line, gt_scatter, gt_x, gt_y, i)

        energy_animate_update(ax_energy, pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te)
        update_phasespace_plot(ps_plot, q, p, i)

    anim = animation.FuncAnimation(fig, animate, frames=q.shape[0], save_count=q.shape[0])

    if show_plot:
        plt.show()

    if save_plot:
        save_animation(anim, config)


def update_pendulum(line, scatter, x, y, i):
    x = x[i, 0]
    y = y[i, 0]

    line.set_data([0, x], [0, y])
    scatter.set_offsets(np.array([x, y]))


def init_pendulum_plot(ax, x, length, xmargin=1.2, ymargin=1.05):
    ax.set_title("Pendulum experiment")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.set_xlim([x.min() * xmargin, x.max() * xmargin])
    ax.set_ylim([-length * ymargin, length * (ymargin - 1)])


def init_pendulum(ax, x, y, color='red', s=500, linewidth=4, zorder=1):
    x = x[0, 0]
    y = y[0, 0]

    lines = ax.plot([0, x], [0, y], linewidth=linewidth, color=color, zorder=zorder)
    scatter = ax.scatter(x, y, color=color, marker='o', s=s, zorder=zorder)

    return lines[0], scatter
