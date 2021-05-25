from matplotlib.lines import Line2D
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Circle
import seaborn as sns
import numpy as np
import os

from hyperverlet.energy import ThreeBodySpringMassEnergy
from hyperverlet.utils.math import calc_dist_2d, calc_theta
from hyperverlet.plotting.energy import plot_energy, init_energy_plot, energy_animate_update

from hyperverlet.plotting.grid_spec import *
from hyperverlet.plotting.utils import plot_spring, set_limits, save_animation, compute_limits, create_gt_pred_legends
from hyperverlet.utils.measures import print_qp_loss
from hyperverlet.utils.misc import load_pickle, format_path


def plot_springs(ax, q, i, colormap=None):
    # Plotted bob circle radius
    r = 0.4
    num_particles = q.shape[1]

    for particle in range(num_particles):
        particle_pos = q[i, particle, :]
        if colormap:
            c0 = Circle(particle_pos[:2], r, fc=colormap[particle], zorder=10)
        else:
            c0 = Circle(particle_pos[:2], r, fc='k', zorder=10)
        ax.add_patch(c0)

        for relative_particle in range(particle + 1, num_particles):
            relative_particle_pos = q[i, relative_particle, :]
            spring_length = calc_dist_2d(particle_pos, relative_particle_pos)
            spring_theta = calc_theta(particle_pos, relative_particle_pos)
            plot_spring(ax, spring_length, theta=spring_theta, xshift=particle_pos[0], yshift=particle_pos[1])


def plot_trail(ax, q, i, trail_len=8, color_map=None):
    # The trail will be divided into trail_len segments and plotted as a fading line.
    for j in range(trail_len):
        imin = i - (trail_len - j)
        if imin < 0:
            continue
        imax = imin + 2
        # The fading looks better if we square the fractional length along the trail.
        alpha = (j / trail_len) ** 2
        for particle in range(q.shape[1]):
            if color_map is not None:
                ax.plot(q[imin:imax, particle, 0], q[imin:imax, particle, 1], c=color_map[particle],
                        solid_capstyle='butt', lw=2, alpha=alpha)
            else:
                ax.plot(q[imin:imax, particle, 0], q[imin:imax, particle, 1], solid_capstyle='butt', lw=2, alpha=alpha)


def three_body_spring_mass_energy_plot(q, p, trajectory, m, k, l, plot_every=1):
    # Detatch and trim data
    q = q.cpu().detach().numpy()[::plot_every]
    p = p.cpu().detach().numpy()[::plot_every]
    trajectory = trajectory.cpu().detach().numpy()[::plot_every]
    m = m.cpu().detach().numpy()
    l = l.cpu().detach().numpy()
    k = k.cpu().detach().numpy()

    # Calculate energy of the system
    energy = ThreeBodySpringMassEnergy()
    ke, pe, te = energy.all_energies(m, q, p, k=k, length=l)

    plot_energy(trajectory, te, ke, pe)


def animate_tbsm(config, show_trail=True, show_springs=False, show_plot=True, cfg=0):
    # Config handler
    plot_every = config["plotting"]["plot_every"]
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)
    save_plot = config["plotting"]["save_plot"]

    print_qp_loss(result_dict["q"], result_dict["p"], result_dict["gt_q"], result_dict["gt_p"], label='total loss')
    print_qp_loss(result_dict["q"][:, cfg], result_dict["p"][:, cfg], result_dict["gt_q"][:, cfg], result_dict["gt_p"][:, cfg], label='cfg loss')

    # Predicted results
    q = result_dict["q"][::plot_every, cfg]
    p = result_dict["p"][::plot_every, cfg]
    trajectory = result_dict["trajectory"][::plot_every, cfg]
    m = result_dict["mass"][cfg]
    l = result_dict["extra_args"]["length"][cfg]
    k = result_dict["extra_args"]["k"][cfg]

    # Ground truth results
    gt_q = result_dict["gt_q"][::plot_every, cfg]
    gt_p = result_dict["gt_p"][::plot_every, cfg]

    # Create grid spec
    fig = plt.figure(figsize=(20, 15))
    ax_experiment, ax_energy, ax_momentum, ax_center_mass, ax_energy_diff = gs_5_3_2(fig)
    #ax_euclid, ax_energy, ax_momentum, *diff_axes = gs_6_4_2(fig)


    # Calculate energy of the system
    energy = ThreeBodySpringMassEnergy()
    ke, pe, te = energy.all_energies(m, q, p, k=k, length=l)
    gt_ke, gt_pe, gt_te = energy.all_energies(m, gt_q, gt_p, k=k, length=l)
    te_max = max(te.max(), gt_te.max())
    diff_ke, diff_pe, diff_te = energy.energy_difference(m, q, p, gt_q, gt_p, k=k, length=l)
    merged_energy_diff = np.concatenate((diff_ke, diff_pe, diff_te))


    # Color maps
    cm = sns.color_palette('Paired', as_cmap=True)
    cm_gt = [cm(i * 2) for i in range(q.shape[1])]
    cm_pred = [cm(i * 2 + 1) for i in range(q.shape[1])]

    # Get x, y coordinate limits
    xlim = gt_q[:, :, 0] if q[:, :, 0].max() < gt_q[:, :, 0].max() else q[:, :, 0]
    ylim = gt_q[:, :, 1] if q[:, :, 1].max() < gt_q[:, :, 1].max() else q[:, :, 1]

    # Compute difference between position
    # gt_pred_q_diff = gt_q - q


    # Compute difference in momentum
    gt_total_p = np.sum(gt_p, axis=1)
    pred_total_p = np.sum(p, axis=1)
    gt_pred_p_diff = gt_total_p - pred_total_p

    # Compute center of mass
    total_mass = np.sum(m * gt_p.shape[1])
    pred_center_mass = np.sum(q * np.expand_dims(m, axis=0), axis=1) / total_mass

    # Initialize plots
    #diff_plots = [(ax, *init_diff_plot(ax, trajectory, gt_pred_q_diff[:, i])) for i, ax in enumerate(diff_axes)]
    p_xplot, p_yplot = init_line_plot(ax_momentum, trajectory, gt_pred_p_diff, title="Predicted momentum difference from ground truth")
    q_xplot, q_yplot = init_line_plot(ax_center_mass, trajectory, pred_center_mass, title="Center of mass")

    gt_pe_plot, gt_ke_plot, gt_te_plot = init_energy_plot(ax_energy, trajectory, gt_te, gt_ke, gt_pe, title="Ground truth energy plot", cm=cm_gt, prefix="GT_")
    pe_plot, ke_plot, te_plot = init_energy_plot(ax_energy, trajectory, te, ke, pe, cm=cm_pred)
    res_pe_plot, res_ke_plot, res_te_plot = init_energy_plot(ax_energy_diff, trajectory, diff_te, diff_ke, diff_pe, title="Energy difference from Ground truth", cm=cm_pred, y_label="Energy difference")

    ax_energy_diff.set_ylim(1.05 * min(merged_energy_diff), max(merged_energy_diff) * 1.05)
    ax_energy.set_ylim(-0.3 * te_max, te_max * 1.05)

    legend_elements = create_gt_pred_legends(q, cm_gt + cm_pred)

    def animate(i):
        ax_experiment.clear()
        ax_experiment.set_aspect('equal')
        ax_experiment.set_title("Three body spring mass experiment")
        set_limits(ax_experiment, xlim, ylim, margin=1.2)

        if show_trail:
            plot_trail(ax_experiment, gt_q, i, color_map=cm_gt, trail_len=5)
            plot_trail(ax_experiment, q, i, color_map=cm_pred, trail_len=5)

        if show_springs:
            plot_springs(ax_experiment, gt_q, i, cm_gt)
            plot_springs(ax_experiment, q, i, cm_pred)

        ax_experiment.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fancybox=True, shadow=True)

        energy_animate_update(ax_energy, gt_pe_plot, gt_ke_plot, gt_te_plot, trajectory, i, gt_pe, gt_ke, gt_te)
        energy_animate_update(ax_energy, pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te)
        energy_animate_update(ax_energy_diff, res_pe_plot, res_ke_plot, res_te_plot, trajectory, i, diff_pe, diff_ke, diff_te)

        animate_lineplot(ax_momentum, p_xplot, p_yplot, trajectory, i, gt_pred_p_diff)
        animate_lineplot(ax_center_mass, q_xplot, q_yplot, trajectory, i, pred_center_mass)

        #for idx, (ax, px, py) in enumerate(diff_plots):
        #    diff_animate_update(ax, px, py, trajectory, i, gt_pred_q_diff[:, idx])

        return []

    anim = animation.FuncAnimation(fig, animate, frames=q.shape[0], repeat=False, interval=1)

    if show_plot:
        plt.show()

    if save_plot:
        save_animation(anim, config)


def tbsm_snapshot(config, cfg=0, slices=6):
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)

    q = result_dict["q"][:, cfg]
    p = result_dict["p"][:, cfg]
    trajectory = result_dict["trajectory"][:, cfg]
    m = result_dict["mass"][cfg]
    l = result_dict["extra_args"]["length"][cfg]
    k = result_dict["extra_args"]["k"][cfg]

    gt_q = result_dict["gt_q"][:, cfg]
    gt_p = result_dict["gt_p"][:, cfg]

    fig = plt.figure(figsize=(25, 10))
    ax_tbsms = gs_line(fig, slices)

    step_size = (q.shape[0] - 1) // (slices - 1)

    # Color maps
    cm = sns.color_palette('Paired', as_cmap=True)
    cm_gt = [cm(i * 2) for i in range(q.shape[1])]
    cm_pred = [cm(i * 2 + 1) for i in range(q.shape[1])]

    xlim = gt_q[:, :, 0] if q[:, :, 0].max() < gt_q[:, :, 0].max() else q[:, :, 0]
    ylim = gt_q[:, :, 1] if q[:, :, 1].max() < gt_q[:, :, 1].max() else q[:, :, 1]

    legend_elements = create_gt_pred_legends(q, cm_gt + cm_pred)

    for idx, (slice, ax_tbsm) in enumerate(zip(range(slices), ax_tbsms)):
        if idx == 0:
            ax_tbsm.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 0), ncol=2, fancybox=True, shadow=True, fontsize=9)
        index = step_size * slice
        label = f"Time {round(trajectory[index])}"

        ax_tbsm.set_aspect('equal')
        set_limits(ax_tbsm, xlim, ylim, margin=1.2)
        ax_tbsm.set_title(label)

        plot_trail(ax_tbsm, gt_q, index, color_map=cm_gt, trail_len=30)
        plot_trail(ax_tbsm, q, index, color_map=cm_pred, trail_len=30)

        plot_springs(ax_tbsm, gt_q, index, cm_gt)
        plot_springs(ax_tbsm, q, index, cm_pred)

    config_name = config["train_args_path"].split('/')[-2]
    solver_name = config["model_args"]["solver"]
    plot_path = f"visualization/{config_name}"
    os.makedirs(plot_path, exist_ok=True)
    filepath = os.path.join(plot_path, solver_name)
    plt.savefig(f'{filepath}.pdf', bbox_inches='tight')
    print(f"Plot saved at {filepath}.pdf")



def init_line_plot(ax, trajectory, line, title=None, x_color='blue', y_color='orange', ylabel="Difference", xlabel="Time"):
    x_plot, = ax.plot(trajectory[0], line[0, 0], color=x_color, label='X')
    y_plot, = ax.plot(trajectory[0], line[0, 1], color=y_color, label='Y')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower left')

    return x_plot, y_plot


def init_diff_energy(ax, trajectory, energy, title=None, color="blue", label=None):
    res_plot, = ax.plot(trajectory[0], energy[0], color=color, label=label)
    ax.set_xlabel(label)
    ax.set_title(title)
    ax.legend(loc='best')
    return res_plot


def animate_lineplot(ax, x_plot, y_plot, trajectory, i, line):
    x_plot.set_data(trajectory[:i + 1], line[:i + 1, 0])
    y_plot.set_data(trajectory[:i + 1], line[:i + 1, 1])
    ax.legend(loc='lower left')

    if i > 0:
        traj_lim = compute_limits(trajectory)
        ax.set_xlim(*traj_lim)

    diff_lim = compute_limits(line)
    ax.set_ylim(*diff_lim)

def animate_energy_diff(ax, energy_diff, trajectory, i, energy):
    energy_diff.set_data(trajectory[:i + 1],)


def example_plot(q, show_trail=True, show_springs=True):
    plt.figure(figsize=(20, 15))
    ax_euclid = plt.gca()
    index = 28
    trail_len = 5

    R = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                  [np.sin(np.pi / 2), np.cos(np.pi / 2)]])
    q = q @ R

    ax_euclid.set_aspect('equal')
    ax_euclid.set_xlabel('X')
    ax_euclid.set_ylabel('Y')
    set_limits(ax_euclid, q[index - trail_len:index + 1, :, 0], q[index - trail_len:index + 1, :, 1], margin=1.2)

    # Color maps
    cm = sns.color_palette('Paired', as_cmap=True)
    cm_pred = [cm(i * 2 + 1) for i in range(q.shape[1])]

    if show_trail:
        plot_trail(ax_euclid, q, index, color_map=cm_pred, trail_len=trail_len)

    if show_springs:
        plot_springs(ax_euclid, q, index, cm_pred)

    os.makedirs('visualization/example', exist_ok=True)
    plt.savefig('visualization/example/three_body_spring_mass.png', bbox_inches='tight')