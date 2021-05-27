import seaborn as sns
import pandas as pd
import os
from matplotlib import pyplot as plt

from hyperverlet.plotting.utils import save_figure
from hyperverlet.utils.measures import qp_mean, valid_prediction_time
from hyperverlet.utils.misc import format_path, load_pickle, load_config


def get_experiment_types(config_path):
    config_split = config_path.split(os.sep)
    exp_type = config_split[-3]
    exp_variation = config_split[-2]
    return exp_type, exp_variation


def generalization_plot(expargs, experiment):
    data = []

    label_mapping = dict(pendulum_length="Length", pendulum_mass="Mass",
                         pendulum_00="$0\%$", pendulum_01="$10\%$", pendulum_02="$20\%$", pendulum_03="$30\%$", pendulum_04="$40\%$",
                         pendulum_not_variable="Pendulum fixed", pendulum_variable="Pendulum random")

    mse_label = "MSE"
    vpt_label = "VPT"
    split_label = "Split"

    sns.set(font_scale=1, rc={'text.usetex': True})

    for idx, args in enumerate(expargs):
        config_path = args.config_path
        config = load_config(config_path)

        result_path = format_path(config, config["result_path"])
        result_dict = load_pickle(result_path)

        exp_type, exp_variation = get_experiment_types(config_path)

        label = label_mapping[exp_variation]

        qp_loss = qp_mean(result_dict["q"], result_dict["p"], result_dict["gt_q"], result_dict["gt_p"])
        vpt_loss = valid_prediction_time(result_dict["q"], result_dict["p"], result_dict["gt_q"], result_dict["gt_p"])
        data.append([label, 'test', qp_loss, vpt_loss])

        if experiment == "generalization_out_of_distribution":
            qp_train_loss = qp_mean(result_dict["train"]["q"], result_dict["train"]["p"], result_dict["train"]["gt_q"], result_dict["train"]["gt_p"])
            vpt_train_loss = valid_prediction_time(result_dict["train"]["q"], result_dict["train"]["p"], result_dict["train"]["gt_q"], result_dict["train"]["gt_p"])

            data.append([label, 'train', qp_train_loss, vpt_train_loss])

    if experiment == "generalization_train_duration":
        x_label = "Step size std. $\%$"

        df = pd.DataFrame(data=data, columns=[x_label, split_label, mse_label, vpt_label])
        sns.lineplot(x=x_label, y=mse_label, data=df)
        save_figure("visualization/generalization", f"{experiment}_mse")
        plt.clf()

        sns.lineplot(x=x_label, y=vpt_label, data=df)
        save_figure("visualization/generalization", f"{experiment}_vpt")
    elif experiment == "generalization_variable_parameters":
        x_label = "Experiments"

        df = pd.DataFrame(data=data, columns=[x_label, split_label, mse_label, vpt_label])
        sns.barplot(x=x_label, y=mse_label, data=df)
        save_figure("visualization/generalization", f"{experiment}_mse")
        plt.clf()

        sns.barplot(x=x_label, y=vpt_label, data=df)
        save_figure("visualization/generalization", f"{experiment}_vpt")
    elif experiment == "generalization_out_of_distribution":
        x_label = " "

        df = pd.DataFrame(data=data, columns=[x_label, split_label, mse_label, vpt_label])
        sns.barplot(x=x_label, y=mse_label, hue=split_label, data=df)
        save_figure("visualization/generalization", f"{experiment}_mse")
        plt.clf()

        df = pd.DataFrame(data=data, columns=[x_label, split_label, mse_label, vpt_label])
        sns.barplot(x=x_label, y=vpt_label, hue=split_label, data=df)
        save_figure("visualization/generalization", f"{experiment}_vpt")