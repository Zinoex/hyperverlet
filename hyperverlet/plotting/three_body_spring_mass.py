from hyperverlet.energy import three_body_spring_mass
from hyperverlet.plotting.energy import plot_energy, init_energy_plot, update_energy_plot
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from hyperverlet.plotting.phasespace import init_phasespace_plot, update_phasespace_plot
from hyperverlet.plotting.spring_mass import calc_theta, calc_dist_2d
from hyperverlet.plotting.utils import plot_spring, set_limits


def three_body_spring_mass_plot(q, p, trajectory, m, k, l, plot_every=1):
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

    # Create grid spec
    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # PLOT - 2: Energy
    init_energy_plot(ax2, trajectory, te, ke, pe)


    # Plotted bob circle radius
    r = 0.02
    num_particles = q.shape[1]
    xlim = q[:, :, 0]
    ylim = q[:, :, 1]
    for i in range(1, q.shape[0]):
        # PLOT - 1: Model
        ax1.clear()

        ax1.set_aspect('equal')
        set_limits(ax1, xlim, ylim)

        for particle in range(num_particles):
            particle_pos = q[i, particle, :]

            c0 = Circle(particle_pos, r, fc='k', zorder=10)
            ax1.add_patch(c0)

            for relative_particle in range(particle + 1, num_particles):
                relative_particle_pos = q[i, relative_particle, :]

                spring_length = calc_dist_2d(particle_pos, relative_particle_pos)
                spring_theta = calc_theta(particle_pos, relative_particle_pos)

                plot_spring(spring_length, ax1, theta=spring_theta, xshift=particle_pos[0], yshift=particle_pos[1])

        # PLOT - 2: Energy
        update_energy_plot(ax2, trajectory, i, te, ke, pe)

        plt.pause(1e-11)


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
