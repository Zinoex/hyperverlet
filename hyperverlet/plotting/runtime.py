import os

import matplotlib.pyplot as plt
import seaborn as sns

from hyperverlet.energy import PendulumEnergy, SpringMassEnergy
from hyperverlet.plotting.utils import fetch_result_dict
from hyperverlet.utils.measures import z_loss


def runtime_plot(config):
    cm = sns.color_palette('muted')
    linestyles = [
         ('dashed', 'dashed'),
         ('dotted', 'dotted'),
         ('dashdot', 'dashdot'),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
         ('solid', 'solid')
    ]

    runtime_config = config['runtime_plot']
    systems = runtime_config['systems']

    for idx, (label, path) in enumerate(config['results'].items()):
        errors = []
        inference_times = []

        for system in systems:
            result_dict = fetch_result_dict(path.format(system=system))

            q = result_dict['q']
            p = result_dict['p']
            gt_q = result_dict['gt_q']
            gt_p = result_dict['gt_p']

            error = z_loss(q, p, gt_q, gt_p).mean()
            errors.append(error)
            inference_times.append(result_dict['inference_time'])

        linestyle = linestyles[idx][1] if runtime_config['linestyles'] else 'solid'
        plt.plot(errors, inference_times, label=label, linewidth=2.0, color=cm[idx], linestyle=linestyle)

    plt.xlabel('MSE')
    plt.ylabel('Runtime [s]')
    plt.xscale('log')
    plt.legend(loc='upper right')

    plot_path = runtime_config['plot_path']
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f'Plot saved at {plot_path}')
