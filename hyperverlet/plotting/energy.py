import matplotlib.pyplot as plt


def update_energy_plot(ax, trajectory, i, te, ke, pe):
    ax.plot([trajectory[i - 1], trajectory[i]], [te[i - 1], te[i]], color='blue')
    ax.plot([trajectory[i - 1], trajectory[i]], [ke[i - 1], ke[i]], color='orange')
    ax.plot([trajectory[i - 1], trajectory[i]], [pe[i - 1], pe[i]], color='green')


def init_energy_plot(ax, trajectory, te, ke, pe):
    ax.scatter(trajectory[0], te[0], label=r'$E_{sys}$')
    ax.scatter(trajectory[0], ke[0], label=r'KE')
    ax.scatter(trajectory[0], pe[0], label=r'PE')
    ax.legend(loc="lower left")


def plot_energy(trajectory, te, ke, pe):
    plt.plot(trajectory, te, color='blue', label=r'$E_{sys}$')
    plt.plot(trajectory, ke, color='orange', label=r'KE')
    plt.plot(trajectory, pe, color='green', label=r'PE')
    plt.legend(loc='lower left')

    plt.show()
