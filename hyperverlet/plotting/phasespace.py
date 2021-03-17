from matplotlib import pyplot as plt


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

    q_half_range_ext = (q.max() - q.min()) * margin / 2
    q_mid = (q.min() + q.max()) / 2

    p_half_range_ext = (p.max() - p.min()) * margin / 2
    p_mid = (p.min() + p.max()) / 2

    ax.set_xlim(q_mid - q_half_range_ext, q_mid + q_half_range_ext)
    ax.set_ylim(p_mid - p_half_range_ext, p_mid + p_half_range_ext)


def update_phasespace_plot(ax, q, p, i):
    ax.plot(q[i], p[i], marker='x', color='black')