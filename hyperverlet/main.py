import torch
import matplotlib.pyplot as plt

from factories.dataset_factory import construct_dataset
from factories.solver_factory import construct_solver
from hyperverlet.experiments import Pendulum, LenardJones, PendulumDataset
from hyperverlet.solvers import HyperEulerSolver, EulerSolver, VelocityVerletSolver, HyperVelocityVerletSolver, \
    StormerVerletSolver, ThirdOrderRuthSolver, FourthOrderRuthSolver
from hyperverlet.models import PendulumModel, LennardJonesMLP
from hyperverlet.test import test
from hyperverlet.timer import timer
from hyperverlet.train import train
from hyperverlet.transforms import Coarsening
from hyperverlet.utils import seed_randomness
from misc.misc import load_config
from plotting.plotting import plot_3d_pos, plot_phasespace, pendulum_plot, plot_2d_pos


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

    pendulum_plot(trajectory, mass, test_dataset.experiment.g, extra_args['length'], q, p, plot_every=1)


if __name__ == '__main__':
    path = '../configurations/template.json'
    main(path)