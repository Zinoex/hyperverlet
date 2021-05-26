import functools
from argparse import ArgumentParser
from dataclasses import dataclass

import torch

from hyperverlet.config_container import preset_config_paths
from hyperverlet.factories.dataset_factory import construct_dataset
from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.lyapunov import lyapunov_solvers_plot
from hyperverlet.plotting.energy import total_energy_plot
from hyperverlet.plotting.generalization import generalization_plot
from hyperverlet.plotting.pendulum import animate_pendulum, pendulum_snapshot
from hyperverlet.plotting.spring_mass import animate_sm, sm_snapshot
from hyperverlet.plotting.three_body_spring_mass import animate_tbsm, tbsm_snapshot
from hyperverlet.plotting.utils import ablation_barplots
from hyperverlet.test import test
from hyperverlet.train import train
from hyperverlet.utils.measures import print_valid_prediction_time, print_qp_mean_loss
from hyperverlet.utils.misc import seed_randomness, load_config, save_pickle, format_path, load_pickle

systems = ['pendulum', 'pendulum20', 'pendulum40', 'pendulum60', 'pendulum80', 'pendulum100',
           'spring_mass', 'spring_mass25', 'spring_mass50', 'spring_mass100', 'spring_mass200',
           'three_body_spring_mass', 'three_body_spring_mass25', 'three_body_spring_mass50', 'three_body_spring_mass100', 'three_body_spring_mass200']


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

    full_parse = commands.add_parser("full", help="Run an evaluation and plotting")
    full_parse.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    full_parse.set_defaults(func=full_run)

    lyapunov_parse = commands.add_parser('lyapunov', help='Create a boxplot showing lyapunov exponent for solvers')
    lyapunov_parse.add_argument('--experiment', type=str, required=True, choices=preset_config_paths.keys(), help="Run experiment on set of predefined json files")
    lyapunov_parse.add_argument('--system', type=str, required=True, choices=systems, help="Name of the system to test")
    lyapunov_parse.set_defaults(func=lyapunov)

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

    full_parse = commands_sequential.add_parser("full", help="Run an evaluation and plotting")
    full_parse.set_defaults(sequential_func=full_run)

    return parser.parse_args()


def replace_system(path, args):
    return ExpArgs(path.format(system=args.system), device=args.device)


def combined(args):
    replace_system_closure = functools.partial(replace_system, args=args)
    expArgs = map(replace_system_closure, preset_config_paths[args.experiment])

    plot_total_energy = False
    plot_ablation = False
    plot_generalization = True

    if plot_total_energy:
        total_energy_plot(expArgs, args.experiment)
    if plot_ablation:
        ablation_barplots(expArgs, args.experiment)
    if plot_generalization:
        generalization_plot(expArgs, args.experiment)


def lyapunov(args):
    replace_system_closure = functools.partial(replace_system, args=args)
    configs = map(replace_system_closure, preset_config_paths[args.experiment])
    lyapunov_solvers_plot(configs)


def sequential(args):
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
    config_path = args.config_path
    config = load_config(config_path)
    dataset = config["dataset_args"]['dataset']

    make_animation = False
    take_snapshot = True
    gather_data = False

    if make_animation:
        animate(config, dataset)
    if take_snapshot:
        snapshot(config, dataset, slices=6)
    if gather_data:
        log_data(config)


def animate(config, dataset):
    plotting_config = config['plotting']
    show_plot = plotting_config['show_plot']

    if dataset == 'pendulum':
        animate_pendulum(config, show_gt=True, show_plot=show_plot)
    elif dataset == 'spring_mass':
        animate_sm(config, show_gt=True, show_plot=show_plot)
    elif dataset == 'three_body_spring_mass':
        animate_tbsm(config, show_trail=True, show_springs=True, show_plot=show_plot)


def log_data(config):
    result_path = format_path(config, config["result_path"])
    result_dict = load_pickle(result_path)

    method = 'qp_loss'

    if method == 'qp_loss':
        print_qp_mean_loss(result_dict["q"], result_dict["p"], result_dict["gt_q"], result_dict["gt_p"], label='')
    elif method == 'vpt':
        print_valid_prediction_time(result_dict["q"], result_dict["p"], result_dict["gt_q"], result_dict["gt_p"], label='')


def snapshot(config, dataset, slices=6):
    if dataset == 'pendulum':
        pendulum_snapshot(config, slices=slices)
    elif dataset == 'spring_mass':
        sm_snapshot(config, slices=slices)
    elif dataset == 'three_body_spring_mass':
        tbsm_snapshot(config, slices=slices, cfg=3)


def full_run(args):
    evaluate(args)
    plot(args)


if __name__ == '__main__':
    args = parse_arguments()
    args.func(args)

