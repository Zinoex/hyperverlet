from hyperverlet.factories.model_factory import construct_model
from hyperverlet.solvers import *


def construct_solver(config):
    dataset_config = config["model_args"]

    solver_name = dataset_config['solver']
    model_path = dataset_config.get("nn_module")

    if model_path is not None:
        hyper_solver = construct_model(model_path)

    solvers = {
        "HyperSolverMixin": HyperSolverMixin,
        "HyperEulerSolver": HyperEulerSolver,
        "EulerSolver": EulerSolver,
        "StormerVerletSolver": StormerVerletSolver,
        "HyperStormerVerletSolver": HyperStormerVerletSolver,
        "VelocityVerletSolver": VelocityVerletSolver,
        "HyperVelocityVerletSolver": HyperVelocityVerletSolver,
        "ThirdOrderRuthSolver": ThirdOrderRuthSolver,
        "FourthOrderRuthSolver": FourthOrderRuthSolver
    }

    solver = solvers[solver_name]

    if solver.trainable:
        return solver(hyper_solver)
    else:
        return solver()
