import datetime
import os

from matplotlib import pyplot as plt
import numpy as np

from hyperverlet.utils.misc import format_path


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


def compute_spring(l, theta=None, xshift=0, yshift=0):
    """Compute the spring from (0,0) to (x,y) as the projection of a helix."""
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

    return xs + xshift, ys + yshift


def plot_spring(ax, l, theta=None, xshift=0, yshift=0):
    """Plot the spring from (0,0) to (x,y)"""
    xs, ys = compute_spring(l, theta, xshift, yshift)

    return ax.plot(xs, ys, c='k', lw=2)[0]


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


def compute_limits(x, margin=1.05):
    x_half_range_ext = (x.max() - x.min()) * margin / 2
    x_mid = (x.min() + x.max()) / 2

    return x_mid - x_half_range_ext, x_mid + x_half_range_ext


def set_limits(ax, x, y, z=None, margin=1.05):
    x_lim = compute_limits(x, margin)
    y_lim = compute_limits(y, margin)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    if z is not None:
        z_lim = compute_limits(z, margin)
        ax.set_zlim(*z_lim)


def save_animation(animation, config):
    plot_path = format_path(config, config["plot_path"])
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    animation.save(plot_path)
    print(f"File saved at {plot_path}")
