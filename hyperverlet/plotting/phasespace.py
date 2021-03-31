from matplotlib import pyplot as plt

from hyperverlet.plotting.utils import set_limits


def plot_phasespace(q, p, show=True):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(p, q, marker='x')

    ax.set_title("Phase space")
    ax.set_xlabel("p")
    ax.set_ylabel("q")

    if show:
        plt.show()


def init_phasespace_plot(ax, q, p, margin=1.05):
    ax.set_title("Phase space")
    ax.set_xlabel("q")
    ax.set_ylabel("p")

    set_limits(ax, q, p, margin=margin)


def update_phasespace_plot(ax, q, p, i):
    ax.plot(q[i], p[i], marker='x', color='black')