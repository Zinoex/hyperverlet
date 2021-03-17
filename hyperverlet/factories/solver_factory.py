from hyperverlet.models.lennard_jones import LennardJonesMLP
from hyperverlet.models.pendulum import PendulumModel
from hyperverlet.models.spring_mass import SpringMassModel
from hyperverlet.solvers import *


def construct_solver(solver_name, hyper_solver=None):
    hyper_solvers = {
        "PendulumModel": PendulumModel(),
        "SpringMassModel": SpringMassModel(),
        "ThreeBodySpringMassModel": SpringMassModel(),
        "LennardJonesMLP": LennardJonesMLP()
    }
    if hyper_solver is not None:
        hyper_solver = hyper_solvers[hyper_solver]

    solvers = {
        "HyperSolverMixin": HyperSolverMixin(),
        "HyperEulerSolver": HyperEulerSolver(hyper_solver),
        "EulerSolver": EulerSolver(),
        "StormerVerletSolver": StormerVerletSolver(),
        "HyperStormerVerletSolver": HyperStormerVerletSolver(hyper_solver),
        "VelocityVerletSolver": VelocityVerletSolver(),
        "HyperVelocityVerletSolver": HyperVelocityVerletSolver(hyper_solver),
        "ThirdOrderRuthSolver": ThirdOrderRuthSolver(),
        "FourthOrderRuthSolver": FourthOrderRuthSolver()
    }
    return solvers[solver_name]
