import torch

from hyperverlet.factories.dataset_factory import construct_dataset
from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.plotting.pendulum import pendulum_plot
from hyperverlet.plotting.spring_mass import spring_mass_plot
from hyperverlet.plotting.three_body_spring_mass import three_body_spring_mass_energy_plot
from hyperverlet.plotting.utils import plot_2d_pos
from hyperverlet.test import test
from hyperverlet.train import train
from hyperverlet.utils import seed_randomness, load_config
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


def main(config_path):
    seed_randomness()
    device = torch.device('cpu')

    config = load_config(config_path)
    dataset_config = config["dataset_args"]

    train_dataset, test_dataset = construct_dataset(dataset_config)

    test_solver = dataset_config['solver']
    hyper_solver = dataset_config.get("hyper_solver")
    solver = construct_solver(test_solver, hyper_solver)

    if solver.trainable:
        train(solver, train_dataset, device)

    q, q_base, p, p_base, mass, trajectory, extra_args = test(solver, test_dataset, device)

    dataset = dataset_config['dataset']
    if dataset == 'pendulum':
        pendulum_plot(q, p, trajectory, mass, test_dataset.experiment.g, extra_args['length'], plot_every=10)
    elif dataset == 'spring_mass':
        spring_mass_plot(q, p, trajectory, mass, extra_args['k'], extra_args['length'], plot_every=1)
    elif dataset == 'three_body_spring_mass':
        three_body_spring_mass_energy_plot(q, p, trajectory, mass, extra_args['k'], extra_args['length'], plot_every=1)
        plot_2d_pos(q, plot_every=100)
        # plot_2d_pos(p, plot_every=100)


if __name__ == '__main__':
    args = parse_arguments()
    main(args.config_path)
