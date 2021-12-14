import os

import matplotlib.pyplot as plt
import seaborn as sns

from hyperverlet.energy import PendulumEnergy, SpringMassEnergy
from hyperverlet.plotting.utils import compute_limits, fetch_result_dict


class EnergyAnimation:
    def __init__(self, ax, trajectory, ke, pe, te, title='Energy plot', cm=None, prefix='', y_label='Energy'):
        if cm is None:
            cm = ['orange', 'green', 'blue']

        self.ke_plot, = ax.plot(trajectory[0], ke[0], label=f'{prefix}KE', color=cm[0])
        self.pe_plot, = ax.plot(trajectory[0], pe[0], label=f'{prefix}PE', color=cm[1])
        self.te_plot, = ax.plot(trajectory[0], te[0], label=f'{prefix}TE', color=cm[2])
        self.ax = ax

        self.trajectory = trajectory
        self.ke, self.pe, self.te = ke, pe, te

        ax.set_xlabel('Time')
        ax.set_ylabel(y_label)
        ax.set_title(title)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.20), fancybox=True, shadow=True, ncol=6)

    def update(self, i):
        self.ke_plot.set_data(self.trajectory[:i + 1], self.ke[:i + 1])
        self.pe_plot.set_data(self.trajectory[:i + 1], self.pe[:i + 1])
        self.te_plot.set_data(self.trajectory[:i + 1], self.te[:i + 1])

        if i > 0:
            self.ax.set_xlim(*compute_limits(self.trajectory))


def energy_plot(config):
    cm = sns.color_palette('muted')
    linestyles = [
         ('dashed', 'dashed'),
         ('dotted', 'dotted'),
         ('dashdot', 'dashdot'),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
         ('solid', 'solid')
    ]

    energy_config = config['energy_plot']
    cfg = energy_config['cfg_idx']

    energy_mapping = dict(pendulum=PendulumEnergy, spring_mass=SpringMassEnergy)
    energy_cls = energy_mapping[config['dataset_args']['dataset']]()
    end = energy_config['end']

    for idx, (label, path) in enumerate(config['results'].items()):
        result_dict = fetch_result_dict(path)

        trajectory = result_dict['trajectory'][:end, cfg]
        mass = result_dict['mass'][cfg]
        extra_args = {k: v[cfg] for k, v in result_dict['extra_args'].items()}

        if energy_config['include_ground_truth'] and idx == 0:
            q = result_dict['gt_q'][:end, cfg]
            p = result_dict['gt_p'][:end, cfg]

            _, _, te = energy_cls.all_energies(mass, q, p, **extra_args)

            linestyle = linestyles[-1][1] if energy_config['linestyles'] else 'solid'
            plt.plot(trajectory, te, label='Ground truth', linewidth=2.0, color=cm[0], linestyle=linestyle)

        q = result_dict['q'][:end, cfg]
        p = result_dict['p'][:end, cfg]

        _, _, te = energy_cls.all_energies(mass, q, p, **extra_args)

        linestyle = linestyles[idx][1] if energy_config['linestyles'] else 'solid'
        plt.plot(trajectory, te, label=label, linewidth=2.0, color=cm[idx + 1 if energy_config['include_ground_truth'] else 0], linestyle=linestyle)

    plt.xlabel('Time')
    plt.ylabel('Total energy')
    plt.legend(loc='lower left')

    plot_path = energy_config['plot_path']
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f'Plot saved at {plot_path}')


def individual_energy_plot(config):
    cm = sns.color_palette('muted')

    energy_config = config['individual_energy_plot']
    cfg = energy_config['cfg_idx']

    energy_mapping = dict(pendulum=PendulumEnergy, spring_mass=SpringMassEnergy)
    energy_cls = energy_mapping[config['dataset_args']['dataset']]()
    end = energy_config['end']

    for idx, (label, path) in enumerate(config['results'].items()):
        result_dict = fetch_result_dict(path)

        q = result_dict['q'][:end, cfg]
        p = result_dict['p'][:end, cfg]
        trajectory = result_dict['trajectory'][:end, cfg]
        mass = result_dict['mass'][cfg]

        extra_args = {k: v[cfg] for k, v in result_dict['extra_args'].items()}

        ke, pe, te = energy_cls.all_energies(mass, q, p, **extra_args)

        plt.plot(trajectory, te, label='Prediction', linewidth=4.0, color=cm[0])

        q = result_dict['gt_q'][:end, cfg]
        p = result_dict['gt_p'][:end, cfg]

        ke, pe, te = energy_cls.all_energies(mass, q, p, **extra_args)

        plt.plot(trajectory, te, label='Ground truth', linewidth=4.0, color=cm[1], linestyle='dashed')

        if 'limits' in energy_config:
            plt.ylim(*energy_config['limits'])

        plt.xlabel('Time')
        plt.ylabel('Total energy')

        plot_path = energy_config['plot_path']
        ext = energy_config['ext']
        os.makedirs(plot_path, exist_ok=True)
        save_path = os.path.join(plot_path, label.replace(' ', '_') + ext)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved at {save_path}")

        plt.clf()

