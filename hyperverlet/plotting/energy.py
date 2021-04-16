import matplotlib.pyplot as plt


def energy_animate_update(ax, pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te):
    pe_plot.set_data(trajectory[:i + 1], pe[:i + 1])
    ke_plot.set_data(trajectory[:i + 1], ke[:i + 1])
    te_plot.set_data(trajectory[:i + 1], te[:i + 1])

    if i > 0:
        traj_range_half = 1.05 * (trajectory[i] - trajectory[0]) / 2
        traj_mid = (trajectory[0] + trajectory[i]) / 2

        ax.set_xlim(traj_mid - traj_range_half, traj_mid + traj_range_half)


def init_energy_plot(ax, trajectory, te, ke, pe, title="Energy plot", cm=['blue', 'orange', 'green'], prefix=""):
    te_plot, = ax.plot(trajectory[0], te[0], label=f'{prefix}TE', color=cm[0])
    ke_plot, = ax.plot(trajectory[0], ke[0], label=f'{prefix}KE', color=cm[1])
    pe_plot, = ax.plot(trajectory[0], pe[0], label=f'{prefix}PE', color=cm[2])
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title(title)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.20), fancybox=True, shadow=True, ncol=6)

    return pe_plot, ke_plot, te_plot


def plot_energy(trajectory, te, ke, pe):
    plt.plot(trajectory, te, color='blue', label=r'$E_{sys}$')
    plt.plot(trajectory, ke, color='orange', label=r'KE')
    plt.plot(trajectory, pe, color='green', label=r'PE')
    plt.legend(loc='lower left')

    plt.show()
