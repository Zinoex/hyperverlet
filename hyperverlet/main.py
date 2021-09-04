import functools
from argparse import ArgumentParser
from dataclasses import dataclass

import torch

from hyperverlet.config_container import preset_config_paths
from hyperverlet.factories.dataset_factory import construct_dataset
from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.plotting.energy import total_energy_plot
from hyperverlet.plotting.pendulum import animate_pendulum, pendulum_snapshot
from hyperverlet.plotting.spring_mass import animate_sm, sm_snapshot
from hyperverlet.test import test
from hyperverlet.train import train
from hyperverlet.utils.measures import print_z_loss
from hyperverlet.utils.misc import seed_randomness, load_config, save_pickle, format_path, load_pickle

systems = ['pendulum20', 'pendulum40', 'pendulum60', 'pendulum80',
           'spring_mass25', 'spring_mass50', 'spring_mass100', 'spring_mass200']


@dataclass
class ExpArgs:
    config_path: str
    device: str


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Select device for tensor operations')

    commands = parser.add_subparsers(help='commands', dest='command')

    evaluate_parse = commands.add_parser("evaluate", help="Test a model")
    evaluate_parse.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    evaluate_parse.set_defaults(func=evaluate)

    plot_parse = commands.add_parser("plot", help="Plot the results")
    plot_parse.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    plot_parse.set_defaults(func=plot)

    result_parse = commands.add_parser('result', help="Print numerical results")
    result_parse.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    result_parse.set_defaults(func=result)

    full_parse = commands.add_parser("full", help="Run an evaluation and plotting")
    full_parse.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    full_parse.set_defaults(func=full_run)

    combined_parse = commands.add_parser('combined', help="Used for plotting multiple configs in the same plot")
    combined_parse.add_argument('--experiment', type=str, required=True, choices=preset_config_paths.keys(), help="Predefined set of json files used for plotting")
    combined_parse.add_argument('--system', type=str, required=True, choices=systems, help="Name of the system to test")
    combined_parse.set_defaults(func=combined)

    sequential_parse = commands.add_parser('sequential', help='Execute evaluate or plot sequentially')
    sequential_parse.add_argument('--experiment', type=str, required=True, choices=preset_config_paths.keys(), help="Run experiment on set of predefined json files")
    sequential_parse.add_argument('--system', type=str, required=True, choices=systems, help="Name of the system to test")
    sequential_parse.set_defaults(func=sequential)

    commands_sequential = sequential_parse.add_subparsers(help='commands', dest='command')

    evaluate_parse = commands_sequential.add_parser("evaluate", help="Test a model")
    evaluate_parse.set_defaults(sequential_func=evaluate)

    plot_parse = commands_sequential.add_parser("plot", help="Plot the results")
    plot_parse.set_defaults(sequential_func=plot)

    result_parse = commands_sequential.add_parser("result", help="Print numerical results")
    result_parse.set_defaults(sequential_func=result)

    full_parse = commands_sequential.add_parser("full", help="Run an evaluation and plotting")
    full_parse.set_defaults(sequential_func=full_run)

    return parser.parse_args()


def replace_system(path, args):
    return ExpArgs(path.format(system=args.system), device=args.device)


def combined(args):
    replace_system_closure = functools.partial(replace_system, args=args)
    exp_args = map(replace_system_closure, preset_config_paths[args.experiment])

    total_energy_plot(exp_args, args.experiment)


def sequential(args):
    for system in ['pendulum20', 'pendulum40', 'pendulum60', 'pendulum80',
           'spring_mass25', 'spring_mass50', 'spring_mass100', 'spring_mass200']:
        args.system = system

        replace_system_closure = functools.partial(replace_system, args=args)
        experiment_args = map(replace_system_closure, preset_config_paths[args.experiment])

        for experiment_arg in experiment_args:
            print('Running: {}'.format(experiment_arg.config_path))
            args.sequential_func(experiment_arg)


def evaluate(args):
    config_path = args.config_path
    seed_randomness()
    config = load_config(config_path)
    device = torch.device(args.device)

    # Solver Construction
    model_config = config['model_args']
    test_solver_name = model_config['solver']

    solver = construct_solver(test_solver_name, model_config["nn_args"]).to(device=device, non_blocking=True)

    # Dataset Construction
    train_dataset, test_dataset = construct_dataset(config, solver.trainable)

    # Train Solver
    if solver.trainable:
        train(solver, train_dataset, device, config)

    # Test Solver
    result_dict = test(solver, train_dataset, test_dataset, device, config)
    result_dict['config'] = config

    save_path = format_path(config, config['result_path'])
    save_pickle(save_path, result_dict)


def plot(args):
    # Idea to update the plotting code:
    # - Restructure such the the plotting config is at focus
    # - Take systems as parameters in plotting config
    # - For animate and snapshot: for-loop over systems
    # - For energy, canonical and MSE bar plots: Combine in a single plot
    # - Remember to take dataset as input in plotting config

    config_path = args.config_path
    config = load_config(config_path)
    dataset = config["dataset_args"]['dataset']

    plotting_types = config['plotting']['plot_types']
    energy_plot = 'energy' in plotting_types
    canonical_plots = 'canonical' in plotting_types

    if 'animation' in plotting_types:
        animate(config, dataset)
    if 'snapshot' in plotting_types:
        snapshot(config, dataset, slices=6)


def animate(config, dataset):
    plotting_config = config['plotting']
    show_plot = plotting_config['show_plot']

    if dataset == 'pendulum':
        animate_pendulum(config, show_gt=True, show_plot=show_plot)
    elif dataset == 'spring_mass':
        animate_sm(config, show_gt=True, show_plot=show_plot)


def snapshot(config, dataset, slices=6):
    if dataset == 'pendulum':
        pendulum_snapshot(config, slices=slices)
    elif dataset == 'spring_mass':
        sm_snapshot(config, slices=slices)


def result(args):
    config = load_config(args.config_path)
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)

    print_z_loss(result_dict["q"], result_dict["p"], result_dict["gt_q"], result_dict["gt_p"], label='')


def full_run(args):
    evaluate(args)
    plot(args)


if __name__ == '__main__':
    args = parse_arguments()
    args.func(args)
