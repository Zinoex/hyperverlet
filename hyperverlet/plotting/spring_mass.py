import sys

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from hyperverlet.energy import SpringMassEnergy
from hyperverlet.plotting.energy import init_energy_plot, plot_energy, energy_animate_update
from hyperverlet.plotting.grid_spec import gs_3_2_3
from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot
from hyperverlet.plotting.utils import plot_spring, save_animation, compute_spring
from hyperverlet.utils.misc import load_pickle, format_path, qp_loss


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

    qp_loss(result_dict["q"], result_dict["p"], result_dict["gt_q"], result_dict["gt_p"], label='total loss')
    qp_loss(result_dict["q"][:, cfg], result_dict["p"][:, cfg], result_dict["gt_q"][:, cfg], result_dict["gt_p"][:, cfg], label='cfg loss')

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
    ax_spring, ax_energy, ax_phase_space = gs_3_2_3(fig)

    # Calculate energy of the system
    energy = SpringMassEnergy()
    ke, pe, te = energy.all_energies(m, q, p, k=k, length=l)
    te_max = float(sys.maxsize) if te.max() == float('inf') else te.max()

    # Initialize plots
    spring_plot = init_sm(ax_spring, q, gt_q, show_gt)
    pe_plot, ke_plot, te_plot = init_energy_plot(ax_energy, trajectory, te, ke, pe)
    ps_plot = init_phasespace_plot(ax_phase_space, q, p)
    ax_energy.set_ylim(-0.3 * te_max, te_max * 1.05)

    def animate(i):
        update_sm(spring_plot, q, gt_q, i, show_gt)
        energy_animate_update(ax_energy, pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te)
        update_phasespace_plot(ps_plot, q, p, i)

    anim = animation.FuncAnimation(fig, animate, frames=q.shape[0], save_count=sys.maxsize)

    if show_plot:
        plt.show()

    if save_plot:
        save_animation(anim, config)


def init_sm(ax, q, gt_q, show_gt, wall_bottom=0.5, wall_top=-0.5, r=0.05):
    ax.set_xlim(-r, np.max(q) * 1.05 + r)
    ax.set_ylim(wall_bottom * 1.05, wall_top * 1.05)
    ax.set_aspect('equal')

    spring = plot_spring(ax, q[0])

    mount = Circle((0, 0), r / 2, fc='k', zorder=10)
    mass = Circle((q[0, 0], 0), r, fc='r', ec='r', zorder=10)
    ax.add_patch(mount)
    ax.add_patch(mass)

    if show_gt:
        gt_bob = Circle((gt_q[0, 0], 0), 0.05 * 0.75, fc='g', ec='g', zorder=11)
        ax.add_patch(gt_bob)
    else:
        gt_bob = None

    return spring, mass, gt_bob


def update_sm(spring_mass, q, gt_q, i, show_gt):
    spring, mass, gt_bob = spring_mass

    xs, ys = compute_spring(q[i])
    spring.set_data(xs, ys)
    mass.set_center((q[i, 0], 0))

    if show_gt:
        gt_bob.set_center((gt_q[i, 0], 0))
