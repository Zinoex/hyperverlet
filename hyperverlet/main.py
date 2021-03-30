from argparse import ArgumentParser

import torch

from hyperverlet.factories.dataset_factory import construct_dataset
from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.plotting.pendulum import pendulum_plot
from hyperverlet.plotting.spring_mass import spring_mass_plot
from hyperverlet.plotting.three_body_spring_mass import three_body_spring_mass_plot
from hyperverlet.test import test
from hyperverlet.train import train
from hyperverlet.utils import seed_randomness, load_config, save_pickle, load_pickle


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


def main(config_path):
    seed_randomness()
    device = torch.device('cpu')

    config = load_config(config_path)

    train_dataset, test_dataset = construct_dataset(config)
    solver = construct_solver(config)

    if solver.trainable:
        train(solver, train_dataset, device)

    if config["train"]:
        result_dict = test(solver, test_dataset, device)
        save_pickle(config["save_path"], result_dict)

    if config["plot"]:
        dataset = config["dataset_args"]['dataset']
        result_dict = load_pickle(config["save_path"])

        if dataset == 'pendulum':
            pendulum_plot(result_dict, test_dataset.experiment.g, plot_every=1)
        elif dataset == 'spring_mass':
            spring_mass_plot(result_dict, plot_every=1)
        elif dataset == 'three_body_spring_mass':
            three_body_spring_mass_plot(result_dict, plot_every=1, show_trail=True, show_springs=True)


if __name__ == '__main__':
    args = parse_arguments()
    main(args.config_path)
