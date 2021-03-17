from matplotlib import pyplot as plt


def plot_3d_pos(q, plot_every=1, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(q[::plot_every, :, 0].flatten(), q[::plot_every, :, 1].flatten(), q[::plot_every, :, 2].flatten(), marker='x')

    ax.set_title("Particle trajectories")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    if show:
        plt.show()


def plot_2d_pos(q, plot_every=1, show=True):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(q[::plot_every, :, 0].flatten(), q[::plot_every, :, 1].flatten(), marker='x')

    ax.set_title("Particle trajectories")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    if show:
        plt.show()