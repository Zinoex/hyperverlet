from matplotlib.lines import Line2D

from hyperverlet.energy import three_body_spring_mass
from hyperverlet.math_utils import calc_dist_2d, calc_theta
from hyperverlet.plotting.energy import plot_energy, init_energy_plot, energy_animate_update
from matplotlib import pyplot as plt, animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import seaborn as sns
import numpy as np

from hyperverlet.plotting.utils import plot_spring, set_limits, save_animation
from hyperverlet.utils import load_pickle


def plot_springs(ax, q, i):
    # Plotted bob circle radius
    r = 0.02
    num_particles = q.shape[1]

    for particle in range(num_particles):
        particle_pos = q[i, particle, :]

        c0 = Circle(particle_pos[:2], r, fc='k', zorder=10)
        ax.add_patch(c0)

        for relative_particle in range(particle + 1, num_particles):
            relative_particle_pos = q[i, relative_particle, :]
            spring_length = calc_dist_2d(particle_pos, relative_particle_pos)
            spring_theta = calc_theta(particle_pos, relative_particle_pos)
            plot_spring(ax, spring_length, theta=spring_theta, xshift=particle_pos[0], yshift=particle_pos[1])


def plot_trail(ax, q, i, trail_len=8, color=None):
    # The trail will be divided into trail_len segments and plotted as a fading line.
    euclidean_dim = q.shape[-1]
    color_map = sns.color_palette("husl", q.shape[1])
    for j in range(trail_len):
        imin = i - (trail_len - j)
        if imin < 0:
            continue
        imax = imin + 2
        # The fading looks better if we square the fractional length along the trail.
        alpha = (j/trail_len) ** 2
        for particle in range(q.shape[1]):
            if euclidean_dim == 2:
                if color is None:
                    ax.plot(q[imin:imax, particle, 0], q[imin:imax, particle, 1], c=color_map[particle], solid_capstyle='butt', lw=2, alpha=alpha)
                else:
                    ax.plot(q[imin:imax, particle, 0], q[imin:imax, particle, 1], c=color, solid_capstyle='butt', lw=2, alpha=alpha)
            elif euclidean_dim == 3:
                ax.plot3D(q[imin:imax, particle, 0], q[imin:imax, particle, 1], q[imin:imax, particle, 2], c=color_map[particle], solid_capstyle='butt', lw=2, alpha=alpha)


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


def animate_tbsm(config, show_trail=True, show_springs=False, show_gt=False, show_plot=True):
    plot_every = config["plotting"]["plot_every"]
    result_dict = load_pickle(config["save_path"])
    save_plot = config["plotting"]["save_plot"]
    dataset = config["dataset_args"]['dataset']

    # Predicted results
    q = result_dict["q"][::plot_every]
    p = result_dict["p"][::plot_every]
    trajectory = result_dict["trajectory"][::plot_every]
    m = result_dict["mass"]
    l = result_dict["extra_args"]["length"]
    k = result_dict["extra_args"]["k"]

    # Ground Truth
    gt_q = np.squeeze(result_dict["gt_q"][::plot_every], axis=1)
    gt_p = np.squeeze(result_dict["gt_p"][::plot_every], axis=1)
    gt_trajectory = result_dict["gt_trajectory"][::plot_every]

    # Create grid spec
    fig = plt.figure(figsize=(20, 15))
    legend_elements = [Line2D([0], [0], color='r', label='Prediction')]

    if show_gt:
        gs = GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[:, :2])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 2])

        # Energy variables
        gt_ke = three_body_spring_mass.calc_kinetic_energy(m, gt_p)
        gt_pe = three_body_spring_mass.calc_potential_energy(k, gt_q, l)
        gt_te = three_body_spring_mass.calc_total_energy(gt_ke, gt_pe)
        gt_pe_plot, gt_ke_plot, gt_te_plot = init_energy_plot(ax3, gt_trajectory, gt_te, gt_ke, gt_pe, title="Ground truth energy plot")
        legend_elements.append(Line2D([0], [0], color='g', label='Ground truth'))
    else:
        gs = GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

    # Get x, y coordinate limits
    xlim = gt_q[:, :, 0] if q[:, :, 0].max() < gt_q[:, :, 0].max() and show_gt else q[:, :, 0]
    ylim = gt_q[:, :, 1] if q[:, :, 1].max() < gt_q[:, :, 1].max() and show_gt else q[:, :, 1]
    zlim = None

    # Calculate energy of the system
    ke = three_body_spring_mass.calc_kinetic_energy(m, p)
    pe = three_body_spring_mass.calc_potential_energy(k, q, l)
    te = three_body_spring_mass.calc_total_energy(ke, pe)

    # Initialize plots
    pe_plot, ke_plot, te_plot = init_energy_plot(ax2, trajectory, te, ke, pe)

    def animate(i):
        ax1.clear()
        ax1.set_aspect('equal')
        ax1.set_title("Three body spring mass experiment")
        set_limits(ax1, xlim, ylim, zlim)

        if show_trail:
            plot_trail(ax1, q, i, color='r', trail_len=15)
            if show_gt:
                plot_trail(ax1, gt_q, i, color='g', trail_len=15)

        if show_springs:
            plot_springs(ax1, q, i)
            if show_gt:
                plot_springs(ax1, gt_q, i)

        ax1.legend(handles=legend_elements, loc='best')

        energy_animate_update(pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te, ax2)
        if show_gt:
            energy_animate_update(gt_pe_plot, gt_ke_plot, gt_te_plot, gt_trajectory, i, gt_pe, gt_ke, gt_te, ax3)

        return []

    anim = animation.FuncAnimation(fig, animate, frames=q.shape[0], repeat=False)

    if show_plot:
        plt.show()

    if save_plot:
        save_animation(dataset, anim)
