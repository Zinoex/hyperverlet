import os.path

from matplotlib import pyplot as plt

from hyperverlet.plotting.utils import set_limits
from hyperverlet.utils.misc import load_config, format_path, load_pickle


def plot_phasespace(config):
    phasespace_config = config['phasespace_plot']
    cfg = phasespace_config['cfg_idx']

    for idx, (label, path) in enumerate(config['results'].items()):
        solver_config = load_config(path)
        result_path = format_path(solver_config, solver_config['result_path'])
        result_dict = load_pickle(result_path)

        q = result_dict['q'][:, cfg]
        p = result_dict['p'][:, cfg]

        plt.scatter(p, q, marker='x')

        plt.title("Phase space")
        plt.xlabel("p")
        plt.ylabel("q")

        plot_path = phasespace_config['plot_path']
        os.makedirs(plot_path, exist_ok=True)
        save_path = os.path.join(plot_path, label.replace(' ', '_') + '.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved at {save_path}")

        plt.clf()


def init_phasespace_plot(ax, q, p, margin=1.05):
    ax.set_title("Phase space")
    ax.set_xlabel("q")
    ax.set_ylabel("p")

    set_limits(ax, q, p, margin=margin)

    points = ax.plot(q[0], p[0], marker='x', color='black')

    return points[0]


def update_phasespace_plot(points, q, p, i):
    points.set_data(q[:i], p[:i])
