from matplotlib.lines import Line2D

from hyperverlet.energy import three_body_spring_mass
from hyperverlet.utils.math import calc_dist_2d, calc_theta
from hyperverlet.plotting.energy import plot_energy, init_energy_plot, energy_animate_update
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Circle

from hyperverlet.plotting.grid_spec import *
from hyperverlet.plotting.utils import plot_spring, set_limits, save_animation
from hyperverlet.utils.misc import load_pickle, format_path


def plot_springs(ax, q, i, colormap=None):
    # Plotted bob circle radius
    r = 0.1
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
    ke = three_body_spring_mass.calc_kinetic_energy(m, p)
    pe = three_body_spring_mass.calc_potential_energy(k, q, l)
    te = three_body_spring_mass.calc_total_energy(ke, pe)

    plot_energy(trajectory, te, ke, pe)


def animate_tbsm(config, show_trail=True, show_springs=False, show_plot=True, cfg=0):
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
    gt_p = result_dict["gt_p"][::plot_every, cfg]

    # Create grid spec
    fig = plt.figure(figsize=(20, 15))
    ax1, ax2, ax3, ax4, ax5 = gs_5_3_2(fig)

    # Calculate energy of the system
    ke = three_body_spring_mass.calc_kinetic_energy(m, p)
    pe = three_body_spring_mass.calc_potential_energy(k, q, l)
    te = three_body_spring_mass.calc_total_energy(ke, pe)

    gt_ke = three_body_spring_mass.calc_kinetic_energy(m, gt_p)
    gt_pe = three_body_spring_mass.calc_potential_energy(k, gt_q, l)
    gt_te = three_body_spring_mass.calc_total_energy(gt_ke, gt_pe)
    te_max = max(te.max(), gt_te.max())

    # Color maps
    cm_gt = ["yellow", "red", "cyan"]
    cm_pred = ["blue", "orange", "green"]


    # Get x, y coordinate limits
    xlim = gt_q[:, :, 0] if q[:, :, 0].max() < gt_q[:, :, 0].max() else q[:, :, 0]
    ylim = gt_q[:, :, 1] if q[:, :, 1].max() < gt_q[:, :, 1].max() else q[:, :, 1]

    gt_pred_diff = gt_q - q

    # Initialize plots
    p1_x, p1_y = init_diff_plot(ax3, trajectory, gt_pred_diff[:, 0])
    p2_x, p2_y = init_diff_plot(ax4, trajectory, gt_pred_diff[:, 1])
    p3_x, p3_y = init_diff_plot(ax5, trajectory, gt_pred_diff[:, 2])

    gt_pe_plot, gt_ke_plot, gt_te_plot = init_energy_plot(ax2, trajectory, gt_te, gt_ke, gt_pe, title="Ground truth energy plot", cm=cm_gt, prefix="GT_")
    pe_plot, ke_plot, te_plot = init_energy_plot(ax2, trajectory, te, ke, pe, cm=cm_pred)
    ax2.set_ylim(-0.3 * te_max, te_max * 1.05)

    legend_elements = create_gt_pred_legends(q, cm_gt + cm_pred)

    def animate(i):
        ax1.clear()
        ax1.set_aspect('equal')
        ax1.set_title("Three body spring mass experiment")
        set_limits(ax1, xlim, ylim, margin=1.2)

        if show_trail:
            plot_trail(ax1, q, i, color_map=cm_pred, trail_len=15)
            plot_trail(ax1, gt_q, i, color_map=cm_gt, trail_len=15)

        if show_springs:
            plot_springs(ax1, q, i, cm_pred)
            plot_springs(ax1, gt_q, i, cm_gt)

        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fancybox=True, shadow=True)

        energy_animate_update(pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te, ax2)
        energy_animate_update(gt_pe_plot, gt_ke_plot, gt_te_plot, trajectory, i, gt_pe, gt_ke, gt_te, ax2)

        diff_animate_update(ax3, p1_x, p1_y, trajectory, i, gt_pred_diff[:, 0])
        diff_animate_update(ax4, p2_x, p2_y, trajectory, i, gt_pred_diff[:, 1])
        diff_animate_update(ax5, p3_x, p3_y, trajectory, i, gt_pred_diff[:, 2])

        return []

    anim = animation.FuncAnimation(fig, animate, frames=q.shape[0], repeat=False)

    if show_plot:
        plt.show()

    if save_plot:
        save_animation(anim, config)


def create_gt_pred_legends(q, cm):
    legend_elements = []

    for idx, color in enumerate(cm):
        if idx >= q.shape[1]:
            label = 'Prediction'
        else:
            label = "Ground truth"
        legend_elements.append(Line2D([0], [0], color=color, label=label))
    return legend_elements


def init_diff_plot(ax, trajectory, diff, title=None, x_color='blue', y_color='orange'):
    x_plot, = ax.plot(trajectory[0], diff[0, 0], color=x_color, label=r'X')
    y_plot, = ax.plot(trajectory[0], diff[0, 1], color=y_color, label=r'Y')
    ax.set_xlabel("Time")
    ax.set_ylabel("Difference")
    ax.set_title(title)
    ax.legend(loc='lower left')

    return x_plot, y_plot


def diff_animate_update(ax, x_plot, y_plot, trajectory, i, diff):
    x_plot.set_data(trajectory[:i + 1], diff[:i + 1, 0])
    y_plot.set_data(trajectory[:i + 1], diff[:i + 1, 1])
    ax.legend(loc='lower left')

    if i > 0:
        traj_range_half = 1.05 * (trajectory[i] - trajectory[0]) / 2
        traj_mid = (trajectory[0] + trajectory[i]) / 2

        ax.set_xlim(traj_mid - traj_range_half, traj_mid + traj_range_half)
    ax.set_ylim(diff.min() * 1.05, diff.max() * 1.05)
