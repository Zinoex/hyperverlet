import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

from hyperverlet.experiments import Pendulum
from hyperverlet.utils.misc import format_path, load_pickle, load_config


def step(x, r):
    return x + r - x ** 2


def lyapunov_solvers_plot(config_paths):
    df_lypunov = pd.DataFrame()

    for config_path in config_paths:
        config = load_config(config_path)
        solver = config['model_args']['solver']
        lambdas = calc_lyapunov_stability(config)
        res_dict = {
            "Solver": [solver] * len(lambdas),
            "Lyapunov exponent": lambdas
        }
        df_res = pd.DataFrame.from_dict(res_dict)
        df_lypunov = df_lypunov.append(df_res, ignore_index=True)

    sns.boxplot(x='Solver', y='Lyapunov exponent', data=df_lypunov)
    plt.show()


def calc_lyapunov_stability(config):
    sns.set_theme(style='whitegrid')

    # Config handler
    plot_every = config["plotting"]["plot_every"]
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)

    # Predicted results
    p = result_dict["p"][::plot_every]
    q = result_dict["q"][::plot_every]
    trajectory = result_dict["trajectory"][::plot_every]
    m = result_dict["mass"]
    l = result_dict["extra_args"]["length"]
    g = result_dict["extra_args"]["g"]

    experiment = Pendulum()

    lambdas = []
    dq = experiment.dq(p, m, trajectory, l)
    dp = experiment.dp(q, m, trajectory, l, g)
    for i in range(trajectory.shape[1]):
        dq_it = dq[:, i]
        dp_it = dp[:, i]
        dqdp = np.concatenate((dq_it, dp_it), axis=0)
        lambdas_ = np.mean(np.log(np.abs(dqdp)))
        lambdas.append(lambdas_)

    return lambdas

    # fig = plt.figure(figsize=(10, 7))
    # ax1 = fig.add_subplot(1, 1, 1)
    # num_x_ticks = len(q)
    # xticks = np.linspace(0, traj_duration, num_x_ticks)
    # zero = [0] * num_x_ticks
    # ax1.plot(xticks, zero, 'g-')
    #
    # ax1.plot(xticks, q, 'r.', alpha=0.3, label="Map")
    # ax1.set_xlabel('r')
    #
    # ax1.plot(rvalues, lambdas, '-b', linewidth=3, label='Lyapunov exponent')
    # ax1.grid('on')
    # ax1.set_xlabel('r')
    # ax1.legend(loc='best')
    # ax1.set_title('Map of x(t+1) = x(t) + r - x(t)^2 versus Lyapunov exponent')
    # plt.show()
