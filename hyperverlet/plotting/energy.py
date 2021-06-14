import os

import matplotlib.pyplot as plt
import seaborn as sns

from hyperverlet.energy import PendulumEnergy, SpringMassEnergy, ThreeBodySpringMassEnergy
from hyperverlet.utils.misc import load_config, format_path, load_pickle


def energy_animate_update(ax, pe_plot, ke_plot, te_plot, trajectory, i, pe, ke, te):
    pe_plot.set_data(trajectory[:i + 1], pe[:i + 1])
    ke_plot.set_data(trajectory[:i + 1], ke[:i + 1])
    te_plot.set_data(trajectory[:i + 1], te[:i + 1])

    if i > 0:
        traj_range_half = 1.05 * (trajectory[i] - trajectory[0]) / 2
        traj_mid = (trajectory[0] + trajectory[i]) / 2

        ax.set_xlim(traj_mid - traj_range_half, traj_mid + traj_range_half)


def init_energy_plot(ax, trajectory, te, ke, pe, title="Energy plot", cm=['blue', 'orange', 'green'], prefix="", y_label="Energy"):
    te_plot, = ax.plot(trajectory[0], te[0], label=f'{prefix}TE', color=cm[0])
    ke_plot, = ax.plot(trajectory[0], ke[0], label=f'{prefix}KE', color=cm[1])
    pe_plot, = ax.plot(trajectory[0], pe[0], label=f'{prefix}PE', color=cm[2])
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.20), fancybox=True, shadow=True, ncol=6)

    return pe_plot, ke_plot, te_plot


def plot_energy(trajectory, te, ke, pe):
    plt.plot(trajectory, te, color='blue', label=r'$E_{sys}$')
    plt.plot(trajectory, ke, color='orange', label=r'KE')
    plt.plot(trajectory, pe, color='green', label=r'PE')
    plt.legend(loc='lower left')

    plt.show()


def total_energy_plot(expargs, experiment, cfg=0):
    cm = sns.color_palette("muted")

    for idx, args in enumerate(expargs):
        config_path = args.config_path
        config = load_config(config_path)

        result_path = format_path(config, config["result_path"])
        result_dict = load_pickle(result_path)

        dataset_config = config["dataset_args"]
        dataset = dataset_config['dataset']

        q = result_dict["q"][:, cfg]
        p = result_dict["p"][:, cfg]
        trajectory = result_dict["trajectory"][:, cfg]
        mass = result_dict["mass"][cfg]

        extra_args = {k: v[cfg] for k, v in result_dict["extra_args"].items()}

        energy_mapping = dict(pendulum=PendulumEnergy, spring_mass=SpringMassEnergy, three_body_spring_mass=ThreeBodySpringMassEnergy)
        energy_cls = energy_mapping[dataset]()
        _, _, te = energy_cls.all_energies(mass, q, p, **extra_args)

        solver = config["model_args"]["solver"]
        label_mapping = dict(FourthOrderRuth="FR4", RungeKutta4="RK4", HyperVelocityVerlet="HyperVerlet", VelocityVerlet="Velocity Verlet")
        label = label_mapping.get(solver, solver)

        linewidth = 1.5
        alpha = 1
        linestyle = "-"

        if solver == "VelocityVerlet":
            linewidth = 0.75
            alpha = 0.5
            linestyle = "-"

        if experiment == "total_energy_full":
            loc = 'best'
            num_samples = trajectory.shape[0]
        else:
            loc = "lower left"
            num_samples = int((trajectory.shape[0] - 1) * 0.4 + 1)

        plt.plot(trajectory[:num_samples], te[:num_samples], label=label, linewidth=linewidth, color=cm[idx], alpha=alpha, linestyle=linestyle)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend(loc=loc)

    plot_path = f"visualization/{experiment}"
    os.makedirs(plot_path, exist_ok=True)
    filepath = os.path.join(plot_path, dataset)
    plt.savefig(f'{filepath}.pdf', bbox_inches='tight')
    print(f"Plot saved at {filepath}.pdf")

