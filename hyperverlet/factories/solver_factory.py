from hyperverlet.factories.model_factory import construct_model
from hyperverlet.solvers import *


def construct_solver(solver_name, nn_args=None):

    if nn_args is not None:
        hyper_solver = construct_model(nn_args)

    solvers = {
        "HyperEuler": HyperEuler,
        "Euler": Euler,
        "HyperHeun": HyperHeun,
        "Heun": Heun,
        "RungeKutta4": RungeKutta4,
        "VelocityVerlet": VelocityVerlet,
        "HyperVelocityVerlet": HyperVelocityVerlet,
        "ThirdOrderRuth": ThirdOrderRuth,
        "FourthOrderRuth": FourthOrderRuth,
        "HyperFourthOrderRuth": HyperFourthOrderRuth,
        "SymplecticHyperVelocityVerlet": SymplecticHyperVelocityVerlet,
        "SympNet": SympNet,
    }

    solver = solvers[solver_name]

    if solver.trainable:
        return solver(hyper_solver)
    else:
        return solver()
