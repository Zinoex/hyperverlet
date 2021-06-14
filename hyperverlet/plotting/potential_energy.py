import numpy as np
import matplotlib.pyplot as plt
import os


def plot_hookes_law(ax_energy, ax_force):
    l = 4
    k = 0.08

    r = np.linspace(0.0001, 10, 1000)
    x = r - l

    pe = 0.5 * k * x ** 2
    force = k * x

    ax_energy.plot(r, pe, label="Hooke's law")
    ax_force.plot(r, force, label="Hooke's law")


def plot_lennard_jones(ax_energy, ax_force):
    sigma = 2
    epsilon = 0.5

    r = np.linspace(0.0001, 10, 1000)

    sigma_r = sigma / r
    sigma_r6 = sigma_r ** 6
    sigma_r12 = sigma_r6 ** 2

    pe = 4 * epsilon * (sigma_r12 - sigma_r6)
    force = -24 * epsilon / r * (2 * sigma_r12 - sigma_r6)

    ax_energy.plot(r, pe, label="Lennard-Jones")
    ax_force.plot(r, force, label="Lennard-Jones")


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4.5))

    plot_hookes_law(*axes)
    plot_lennard_jones(*axes)

    axes[0].set_ylim(-0.6, 1.2)
    axes[1].set_ylim(-1, 1)

    axes[0].legend()
    axes[1].legend()

    axes[0].axhline(linestyle='--', c='k')
    axes[1].axhline(linestyle='--', c='k')

    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Energy')

    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Force')

    #plt.show()
    os.makedirs('visualization/example', exist_ok=True)
    fig.savefig('visualization/example/potential_energy.pdf', bbox_inches='tight')
