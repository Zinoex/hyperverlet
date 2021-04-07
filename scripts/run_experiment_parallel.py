from argparse import ArgumentParser
from multiprocessing import Pool

from hyperverlet.main import evaluate, plot, full_run

num_cores = 8

systems = ['pendulum', 'spring_mass', 'three_body_spring_mass']
config_paths = {
    'integer_comparison': [
        'configurations/integrator_experiments/{system}/euler.json'
        'configurations/integrator_experiments/{system}/heun.json'
        'configurations/integrator_experiments/{system}/hypereuler.json'
        'configurations/integrator_experiments/{system}/velocityverlet.json'
        'configurations/integrator_experiments/{system}/hyperverlet.json'
        'configurations/integrator_experiments/{system}/hyperheun.json'
        'configurations/integrator_experiments/{system}/rk4.json'
        'configurations/integrator_experiments/{system}/ruth4.json'
    ]
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--experiment', type=str, choices=config_paths.keys(), help="Name of the experiment to run")
    parser.add_argument('--system', type=str, choices=systems, help="Name of the system to test")

    commands = parser.add_subparsers(help='commands', dest='command')

    evaluate_parse = commands.add_parser("evaluate", help="Test a model")
    evaluate_parse.set_defaults(func=evaluate)

    plot_parse = commands.add_parser("plot", help="Plot the results")
    plot_parse.set_defaults(func=plot)

    full_parse = commands.add_parser("full", help="Run an evaluation and plotting")
    full_parse.set_defaults(func=full_run)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    def replace_system(path):
        return path.format(system=args.system)

    experiment_config_paths = map(replace_system, config_paths[args.experiment])

    with Pool(num_cores - 1) as p:
        p.map(args.func, experiment_config_paths)
