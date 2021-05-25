import os

import numpy as np
from matplotlib import pyplot as plt, animation

from hyperverlet.energy import PendulumEnergy
from hyperverlet.plotting.energy import init_energy_plot, plot_energy, energy_animate_update
from hyperverlet.plotting.grid_spec import gs_3_2_3, gs_line
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot
from hyperverlet.plotting.utils import save_animation, create_gt_pred_legends
from hyperverlet.utils.misc import load_pickle, format_path, qp_loss


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

    qp_loss(result_dict["q"], result_dict["p"], result_dict["gt_q"], result_dict["gt_p"], label='total loss')
    qp_loss(result_dict["q"][:, cfg], result_dict["p"][:, cfg], result_dict["gt_q"][:, cfg], result_dict["gt_p"][:, cfg], label='cfg loss')

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
    ax_energy.set_ylim(-0.3 * te.max(), te.max() * 1.05)

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


def pendulum_snapshot(config, cfg=0, slices=6):
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)

    q = result_dict["q"][:, cfg]
    gt_q = result_dict["gt_q"][:, cfg]
    length = result_dict["extra_args"]["length"][cfg]
    trajectory = result_dict["trajectory"][:, cfg]

    x = length * np.sin(q)
    y = -length * np.cos(q)
    gt_x = length * np.sin(gt_q)
    gt_y = -length * np.cos(gt_q)

    fig = plt.figure(figsize=(20, 15))
    ax_pendulums = gs_line(fig, slices)

    step_size = (q.shape[0] - 1) // (slices - 1)
    cm = ["green", "red"]
    legend_handles = create_gt_pred_legends(q, cm)

    for idx, (slice, ax_pendulum) in enumerate(zip(range(slices), ax_pendulums)):
        if idx == 0:
            ax_pendulum.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0, 1), ncol=1, fancybox=True, shadow=True)
        index = step_size * slice
        label = f"Time {int(trajectory[index])}"

        init_pendulum_plot(ax_pendulum, x, length, title=label, xmargin=0.25, ymargin=0.25, set_ylabel=idx==0)
        line, scatter = init_pendulum(ax_pendulum, x, y, color=cm[1], zorder=1)
        gt_line, gt_scatter = init_pendulum(ax_pendulum, gt_x, gt_y, color=cm[0], s=300, linewidth=2, zorder=2)

        update_pendulum(line, scatter, x, y, index)
        update_pendulum(gt_line, gt_scatter, gt_x, gt_y, index)

    config_name = config["train_args_path"].split('/')[-2]
    solver_name = config["model_args"]["solver"]
    plot_path = f"visualization/{config_name}"
    os.makedirs(plot_path, exist_ok=True)
    filepath = os.path.join(plot_path, solver_name)
    plt.savefig(f'{filepath}.pdf', bbox_inches='tight')
    print(f"Plot saved at {filepath}.pdf")


def update_pendulum(line, scatter, x, y, i):
    x = x[i, 0]
    y = y[i, 0]

    line.set_data([0, x], [0, y])
    scatter.set_offsets(np.array([x, y]))


def init_pendulum_plot(ax, x, length, xmargin=1.2, ymargin=1.05, title="Pendulum experiment", set_ylabel=True):
    ax.set_title(title)
    ax.set_xlabel('X')
    if set_ylabel:
        ax.set_ylabel('Y')
    else:
        ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.set_xlim([-length - xmargin, length + xmargin])
    ax.set_ylim([-length - ymargin, length + ymargin])


def init_pendulum_animate(ax, x, length, xmargin=1.2, ymargin=1.05, title="Pendulum experiment"):
    ax.set_title(title)
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
