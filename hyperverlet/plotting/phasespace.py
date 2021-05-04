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

    points = ax.plot(q[0], p[0], marker='x', color='black')

    return points[0]


def update_phasespace_plot(points, q, p, i):
    points.set_data(q[:i], p[:i])