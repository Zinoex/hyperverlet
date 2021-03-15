import torch

from hyperverlet.factories.dataset_factory import construct_dataset
from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.test import test
from hyperverlet.train import train
from hyperverlet.utils import seed_randomness, load_config
from hyperverlet.plotting.plotting import pendulum_plot


def main(config_path):
    # Project initialization
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


if __name__ == '__main__':
    path = '../configurations/pendulum/velocityverlet.json'
    main(path)
