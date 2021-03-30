from argparse import ArgumentParser

import torch

from hyperverlet.factories.dataset_factory import construct_dataset
from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.plotting.pendulum import pendulum_plot
from hyperverlet.plotting.spring_mass import spring_mass_plot
from hyperverlet.plotting.three_body_spring_mass import three_body_spring_mass_plot
from hyperverlet.test import test
from hyperverlet.train import train
from hyperverlet.utils import seed_randomness, load_config


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

    q, q_base, p, p_base, mass, trajectory, extra_args = test(solver, test_dataset, device)

    dataset = config["dataset_args"]['dataset']

    if dataset == 'pendulum':
        pendulum_plot(q, p, trajectory, mass, test_dataset.experiment.g, extra_args['length'], plot_every=10)
    elif dataset == 'spring_mass':
        spring_mass_plot(q, p, trajectory, mass, extra_args['k'], extra_args['length'], plot_every=1)
    elif dataset == 'three_body_spring_mass':
        three_body_spring_mass_plot(q, p, trajectory, mass, extra_args['k'], extra_args['length'], plot_every=30, show_trail=True, show_springs=True)
        #three_body_spring_mass_energy_plot(q, p, trajectory, mass, extra_args['k'], extra_args['length'], plot_every=1)
        #plot_2d_pos(q, plot_every=100)


if __name__ == '__main__':
    args = parse_arguments()
    main(args.config_path)
