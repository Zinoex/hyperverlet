from matplotlib import pyplot as plt
import numpy as np


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


def plot_spring(ax, l, theta=None, xshift=0, yshift=0):
    """Plot the spring from (0,0) to (x,y) as the projection of a helix."""
    if theta is None:
        theta = np.pi / 2

    # Spring turn radius, number of turns
    rs, ns = 0.02, 25
    # Number of data points for the helix
    Ns = 1000
    # We don't draw coils all the way to the end of the pendulum:
    # pad a bit from the anchor and from the bob by these number of points
    ipad1, ipad2 = 150, 150
    w = np.linspace(0, l, Ns).flatten()
    # Set up the helix along the x-axis ...
    xp = np.zeros(Ns)
    xp[ipad1:-ipad2] = rs * np.sin(2 * np.pi * ns * w[ipad1:-ipad2] / l)
    # ... then rotate it to align with  the pendulum and plot.
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    xs, ys = - R @ np.vstack((xp, w))
    ax.plot(xs + xshift, ys + yshift, c='k', lw=2)


def plot_spring_3d(ax, l, theta_x=0, theta_y=0, xshift=0, yshift=0, zshift=0):
    # Spring turn radius, number of turns
    rs, ns = 0.02, 25
    # Number of data points for the helix
    Ns = 1000
    # We don't draw coils all the way to the end of the pendulum:
    # pad a bit from the anchor and from the bob by these number of points
    ipad1, ipad2 = 150, 150
    w = np.linspace(0, l, Ns).flatten()
    # Set up the helix along the x-axis ...
    xp = np.zeros(Ns)
    xp[ipad1:-ipad2] = rs * np.sin(2 * np.pi * ns * w[ipad1:-ipad2] / l)

    yp = np.zeros(Ns)
    yp[ipad1:-ipad2] = rs * np.cos(2 * np.pi * ns * w[ipad1:-ipad2] / l)
    # ... then rotate it to align with  the pendulum and plot.
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])

    xs, ys, zs = Rx @ Ry @ np.vstack((xp, yp, w))
    ax.plot3D(xs + xshift, ys + yshift, zs + zshift, c='k', lw=2)


def set_limits(ax, x, y, z=None, margin=1.05):
    x_half_range_ext = (x.max() - x.min()) * margin / 2
    x_mid = (x.min() + x.max()) / 2

    y_half_range_ext = (y.max() - y.min()) * margin / 2
    y_mid = (y.min() + y.max()) / 2

    ax.set_xlim(x_mid - x_half_range_ext, x_mid + x_half_range_ext)
    ax.set_ylim(y_mid - y_half_range_ext, y_mid + y_half_range_ext)

    if z is not None:
        z_half_range_ext = (z.max() - z.min()) * margin / 2
        z_mid = (z.min() + z.max()) / 2
        ax.set_zlim(z_mid - z_half_range_ext, z_mid + z_half_range_ext)