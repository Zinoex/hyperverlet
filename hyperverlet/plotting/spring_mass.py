import sys

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from hyperverlet.energy import SpringMassEnergy
from hyperverlet.plotting.energy import init_energy_plot, plot_energy, energy_animate_update
from hyperverlet.plotting.grid_spec import gs_3_2_3
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot
from hyperverlet.plotting.utils import plot_spring, save_animation
from hyperverlet.utils.misc import load_pickle, format_path


def spring_mass_energy_plot(q, p, trajectory, m, k, l, plot_every=1):
    # Detatch and trim data
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    trajectory = trajectory.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()
    k = k.cpu().detach().numpy()

    # Calculate energy of the system
    energy = SpringMassEnergy()
    ke, pe, te = energy.all_energies(m, q, p, k=k, length=l)

    plot_energy(trajectory, te, ke, pe)


def animate_sm(config, show_gt=False, show_plot=True, cfg=0):
    plot_every = config["plotting"]["plot_every"]
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)
    save_plot = config["plotting"]["save_plot"]

    # Predicted results
    q = result_dict["q"][::plot_every, cfg]
    p = result_dict["p"][::plot_every, cfg]
    trajectory = result_dict["trajectory"][::plot_every, cfg]
    m = result_dict["mass"][cfg]
    l = result_dict["extra_args"]["length"][cfg]
    k = result_dict["extra_args"]["k"][cfg]

    # Ground Truth
    gt_q = result_dict["gt_q"][::plot_every, cfg]

    # Create grid spec
    fig = plt.figure(figsize=(20, 15))
    ax1, ax2, ax3 = gs_3_2_3(fig)

    # Calculate energy of the system
    energy = SpringMassEnergy()
    ke, pe, te = energy.all_energies(m, q, p, k=k, length=l)

    # Initialize plots
    pe_plot, ke_plot, te_plot = init_energy_plot(ax2, trajectory, te, ke, pe)
    ps_plot = init_phasespace_plot(ax3, q, p)

    def animate(i):
        ax1.clear()
        update_sm(ax1, q, i)
        if show_gt:
            gt_c1 = Circle((gt_q[i, 0], 0), 0.05*0.75, fc='g', ec='g', zorder=10)
            ax1.add_patch(gt_c1)

        energy_animate_update(ax2, pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te)
        update_phasespace_plot(ps_plot, q, p, i)

    anim = animation.FuncAnimation(fig, animate, frames=q.shape[0], save_count=sys.maxsize)

    if show_plot:
        plt.show()

    if save_plot:
        save_animation(anim, config)


def update_sm(ax, q, i, wall_bottom=0.5, wall_top=-0.5, r=0.05):
    ax.set_xlim(-r, np.max(q) * 1.05 + r)
    ax.set_ylim(wall_bottom * 1.05, wall_top * 1.05)
    ax.set_aspect('equal')

    plot_spring(ax, q[i])

    c0 = Circle((0, 0), r / 2, fc='k', zorder=10)
    c1 = Circle((q[i, 0], 0), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
