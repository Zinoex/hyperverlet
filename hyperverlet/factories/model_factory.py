from hyperverlet.models.lennard_jones import LennardJonesMLP
from hyperverlet.models.pendulum import *
from hyperverlet.models.spring_mass import SpringMassModel, SymplecticSpringMassModel
from hyperverlet.models.symplectic import LASymplecticModel
from hyperverlet.models.three_body_gravity import ThreeBodyGravityModel
from hyperverlet.models.three_body_spring_mass import ThreeBodySpringMassModel, ThreeBodySpringMassGraphModel


def construct_model(module_config):
    module = module_config["module"]

    module_mapping = dict(
        PendulumModel=PendulumModel,
        SymplecticPendulumModel=SymplecticPendulumModel,
        PendulumSharedModel=PendulumSharedModel,
        CurvaturePendulumModel=CurvaturePendulumModel,
        PostPendulumModel=PostPendulumModel,
        StatePostPendulumModel=StatePostPendulumModel,
        PrePostPendulumModel=PrePostPendulumModel,
        TimePostPendulumModel=TimePostPendulumModel,
        SpringMassModel=SpringMassModel,
        ThreeBodySpringMassModel=ThreeBodySpringMassModel,
        ThreeBodySpringMassGraphModel=ThreeBodySpringMassGraphModel,
        ThreeBodyGravityModel=ThreeBodyGravityModel,
        LennardJonesMLP=LennardJonesMLP,
        SymplecticSpringMassModel=SymplecticSpringMassModel,
        LASymplecticModel=LASymplecticModel
    )

    module = module_mapping[module]
    model = module(module_config)

    return model
