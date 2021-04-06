from hyperverlet.factories.model_factory import construct_model
from hyperverlet.solvers import *


def construct_solver(solver_name, nn_args=None):

    if nn_args is not None:
        hyper_solver = construct_model(nn_args)

    solvers = {
        "HyperEulerSolver": HyperEulerSolver,
        "EulerSolver": EulerSolver,
        "HyperHeunSolver": HyperHeunSolver,
        "HeunSolver": HeunSolver,
        "RungeKutta4Solver": RungeKutta4Solver,
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
