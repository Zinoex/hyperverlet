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

    commands = parser.add_subparsers(help='commands', dest='command')

    evaluate_parse = commands.add_parser("evaluate", help="Test a model")
    evaluate_parse.set_defaults(func=evaluate)

    plot_parse = commands.add_parser("plot", help="Test a model")
    plot_parse.set_defaults(func=plot)

    return parser.parse_args()


def evaluate(config_path):
    seed_randomness()
    config = load_config(config_path)
    device = torch.device('cpu')

    # Dataset Construction
    train_dataset, test_dataset = construct_dataset(config)

    # Solver Construction
    model_config = config['model_args']
    test_solver_name = model_config['solver']
    model_path = model_config.get("nn_module")
    solver = construct_solver(test_solver_name, model_path)

    # Train Solver
    if solver.trainable:
        train(solver, train_dataset, device, config)

    # Test Solver
    result_dict = test(solver, test_dataset, device)
    save_pickle(config["save_path"], result_dict)


def plot(config_path):
    config = load_config(config_path)
    dataset = config["dataset_args"]['dataset']
    plot_every = config["plotting"]["plot_every"]
    result_dict = load_pickle(config["save_path"])

    if dataset == 'pendulum':
        pendulum_plot(result_dict, plot_every=plot_every)
    elif dataset == 'spring_mass':
        spring_mass_plot(result_dict, plot_every=plot_every)
    elif dataset == 'three_body_spring_mass':
        three_body_spring_mass_plot(result_dict, plot_every=plot_every, show_trail=True, show_springs=True)


if __name__ == '__main__':
    args = parse_arguments()
    args.func(args.config_path)