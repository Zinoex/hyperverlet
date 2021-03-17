
def update_energy_plot(ax, time, i, te, ke, pe):
    ax.plot([time[i - 1], time[i]], [te[i - 1], te[i]], color='blue')
    ax.plot([time[i - 1], time[i]], [ke[i - 1], ke[i]], color='orange')
    ax.plot([time[i - 1], time[i]], [pe[i - 1], pe[i]], color='green')


def init_energy_plot(ax, time, te, ke, pe):
    ax.scatter(time[0], te[0], label=r'$E_{sys}$')
    ax.scatter(time[0], ke[0], label=r'KE')
    ax.scatter(time[0], pe[0], label=r'PE')
    ax.legend(loc="lower left")