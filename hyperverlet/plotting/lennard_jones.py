import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from hyperverlet.plotting.energy import init_energy_plot, update_energy_plot


def lj_plot(time: np.array, pe: np.array, ke: np.array, te: np.array, dist: np.array, eps, sigma):
    fig = plt.figure(figsize=(80, 60))
    gs = GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # PLOT - 2: Energy
    init_energy_plot(ax2, time, te, ke, pe)

    for i in range(1, len(time)):
        # PLOT - 1: Model
        ax1.clear()

        ax1.scatter(dist[i], 0, color='blue', marker='o', s=500, alpha=0.8)
        ax1.set_xlim([0, 3])
        ax1.set_xlabel('X', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Y', fontweight='bold', fontsize=14)
        ax1.grid()

        # Potential energy curve
        r = np.linspace(0.8, 3, 100)
        U = np.array(eps * ((sigma / r) ** 12 - 2 * (sigma / r) ** 6))
        ax1.plot(r, U, 'r', label='LJ potential')
        ax1.legend()

        # PLOT - 2: Energy
        update_energy_plot(ax2, time, i, te, ke, pe)

        plt.pause(1E-11)