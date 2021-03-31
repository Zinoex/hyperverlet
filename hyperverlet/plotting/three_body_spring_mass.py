from mpl_toolkits.mplot3d import art3d

from hyperverlet.energy import three_body_spring_mass
from hyperverlet.plotting.energy import plot_energy, init_energy_plot, update_energy_plot
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import seaborn as sns

from hyperverlet.plotting.spring_mass import calc_theta, calc_dist_2d
from hyperverlet.plotting.utils import plot_spring, set_limits, plot_spring_3d


def three_body_spring_mass_plot(result_dict, plot_every=1, show_trail=True, show_springs=False):
    q = result_dict["q"][::plot_every]
    p = result_dict["p"][::plot_every]
    trajectory = result_dict["trajectory"][::plot_every]
    m = result_dict["mass"]
    l = result_dict["extra_args"]["length"]
    k = result_dict["extra_args"]["k"]

    euclidean_dim = q.shape[-1]

    # Calculate energy of the system
    ke = three_body_spring_mass.calc_kinetic_energy(m, p)
    pe = three_body_spring_mass.calc_potential_energy(k, q, l)
    te = three_body_spring_mass.calc_total_energy(ke, pe)

    # Create grid spec
    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(1, 2)

    # Get x, y coordinate limits
    xlim = q[:, :, 0]
    ylim = q[:, :, 1]
    zlim = None

    if euclidean_dim == 2:
        ax1 = fig.add_subplot(gs[0, 0])
    elif euclidean_dim == 3:
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        zlim = q[:, :, 2]

    ax2 = fig.add_subplot(gs[0, 1])

    # PLOT - 2: Energy
    init_energy_plot(ax2, trajectory, te, ke, pe)

    for i in range(1, q.shape[0]):
        # PLOT - 1: Model
        ax1.clear()
        if euclidean_dim == 2:
            ax1.set_aspect('equal')
        set_limits(ax1, xlim, ylim, zlim)

        if show_trail:
            plot_trail(ax1, q, i)

        if show_springs:
            plot_springs(ax1, q, i)

        # PLOT - 2: Energy
        update_energy_plot(ax2, trajectory, i, te, ke, pe)
        plt.pause(1e-20)


def plot_springs(ax, q, i):
    euclidean_dim = q.shape[-1]
    # Plotted bob circle radius
    r = 0.02
    num_particles = q.shape[1]

    for particle in range(num_particles):
        particle_pos = q[i, particle, :]

        c0 = Circle(particle_pos[:2], r, fc='k', zorder=10)
        ax.add_patch(c0)
        if euclidean_dim == 3:
            art3d.pathpatch_2d_to_3d(c0, z=particle_pos[-1])

        for relative_particle in range(particle + 1, num_particles):
            relative_particle_pos = q[i, relative_particle, :]
            spring_length = calc_dist_2d(particle_pos, relative_particle_pos)

            if euclidean_dim == 2:
                spring_theta = calc_theta(particle_pos, relative_particle_pos)
                plot_spring(ax, spring_length, theta=spring_theta, xshift=particle_pos[0], yshift=particle_pos[1])


def plot_trail(ax, q, i, trail_len=8):
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
                ax.plot(q[imin:imax, particle, 0], q[imin:imax, particle, 1], c=color_map[particle], solid_capstyle='butt', lw=2, alpha=alpha)
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
