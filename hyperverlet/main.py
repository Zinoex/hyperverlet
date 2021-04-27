from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from multiprocessing import Pool

import torch

from hyperverlet.factories.dataset_factory import construct_dataset
from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.lyapunov import lyapunov_solvers_plot
from hyperverlet.plotting.pendulum import animate_pendulum
from hyperverlet.plotting.spring_mass import animate_sm
from hyperverlet.plotting.three_body_spring_mass import animate_tbsm
from hyperverlet.test import test
from hyperverlet.train import train
from hyperverlet.utils.misc import seed_randomness, load_config, save_pickle, format_path

systems = ['pendulum', 'spring_mass', 'three_body_spring_mass']
config_paths = {
    'integrator_comparison': [
        'configurations/integrator_comparison/{system}/euler.json',
        'configurations/integrator_comparison/{system}/heun.json',
        'configurations/integrator_comparison/{system}/hypereuler.json',
        'configurations/integrator_comparison/{system}/velocityverlet.json',
        'configurations/integrator_comparison/{system}/hyperverlet.json',
        'configurations/integrator_comparison/{system}/hyperheun.json',
        'configurations/integrator_comparison/{system}/rk4.json',
        'configurations/integrator_comparison/{system}/ruth4.json'
    ]
}


def parse_arguments():
    parser = ArgumentParser()

    commands = parser.add_subparsers(help='commands', dest='command')

    evaluate_parse = commands.add_parser("evaluate", help="Test a model")
    evaluate_parse.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    evaluate_parse.set_defaults(func=evaluate)

    experiment_parse = commands.add_parser("experiment", help="Run a full experiment")
    experiment_parse.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    experiment_parse.set_defaults(func=experiment_parse)

    plot_parse = commands.add_parser("plot", help="Plot the results")
    plot_parse.add_argument('--config-path', type=str, required=True, help="Path to the configuration file")
    plot_parse.set_defaults(func=plot)

    full_parse = commands.add_parser("full", help="Run an evaluation and plotting")
    full_parse.set_defaults(func=full_run)

    lyapunov_parse = commands.add_parser('lyapunov', help='Create a boxplot showing lyapunov exponent for solvers')
    lyapunov_parse.add_argument('--experiment', type=str, required=True, choices=config_paths.keys(), help="Run experiment on set of predfefined json files")
    lyapunov_parse.add_argument('--system', type=str, required=True, choices=systems, help="Name of the system to test")
    lyapunov_parse.set_defaults(func=lyapunov)

    parallel_parse = commands.add_parser('parallel', help='Execute evaluate or plot in parallel')
    parallel_parse.add_argument('--experiment', type=str, required=True, choices=config_paths.keys(), help="Run experiment on set of predfefined json files")
    parallel_parse.add_argument('--system', type=str, required=True, choices=systems, help="Name of the system to test")
    parallel_parse.add_argument('--num-processes', type=int, default=7, help="Number of parallel processes")
    parallel_parse.set_defaults(func=parallel)

    commands_parallel = parallel_parse.add_subparsers(help='commands', dest='command')

    evaluate_parse = commands_parallel.add_parser("evaluate", help="Test a model")
    evaluate_parse.set_defaults(parallel_func=evaluate)

    plot_parse = commands_parallel.add_parser("plot", help="Plot the results")
    plot_parse.set_defaults(parallel_func=plot)

    full_parse = commands_parallel.add_parser("full", help="Run an evaluation and plotting")
    full_parse.set_defaults(parallel_func=full_run)

    return parser.parse_args()


def lyapunov(args):
    def replace_system(path):
        return path.format(system=args.system)

    configs = map(replace_system, config_paths[args.experiment])
    lyapunov_solvers_plot(configs)


def parallel(args):
    ExpArgs = namedtuple('ExpArgs', ['config_path'])

    def replace_system(path):
        return ExpArgs(path.format(system=args.system))

    experiment_config_paths = map(replace_system, config_paths[args.experiment])

    with Pool(args.num_processes) as p:
        p.map(args.parallel_func, experiment_config_paths)


def evaluate(args):
    config_path = args.config_path
    seed_randomness()
    config = load_config(config_path)
    device = torch.device('cpu')

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
    result_dict = test(solver, test_dataset, device, config)
    result_dict['config'] = config

    save_path = format_path(config, config['result_path'])
    save_pickle(save_path, result_dict)


def plot(args):
    config_path = args.config_path
    config = load_config(config_path)
    dataset = config["dataset_args"]['dataset']

    if dataset == 'pendulum':
        animate_pendulum(config, show_gt=True)
    elif dataset == 'spring_mass':
        animate_sm(config, show_gt=True)
    elif dataset == 'three_body_spring_mass':
        animate_tbsm(config, show_trail=True, show_springs=True, show_plot=True)


def full_run(args):
    evaluate(args)
    plot(args)


if __name__ == '__main__':
    args = parse_arguments()
    args.func(args)
