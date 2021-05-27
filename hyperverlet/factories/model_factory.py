from hyperverlet.models.lennard_jones import LennardJonesMLP
from hyperverlet.models.pendulum import PendulumModel, PendulumSharedModel, PostPendulumModel, StatePostPendulumModel, \
    PrePostPendulumModel, TimePostPendulumModel, CurvaturePendulumModel
from hyperverlet.models.spring_mass import SpringMassModel
from hyperverlet.models.three_body_spring_mass import ThreeBodySpringMassModel, ThreeBodySpringMassGraphModel


def construct_model(module_config):
    module = module_config["module"]

    module_mapping = dict(
        PendulumModel=PendulumModel,
        PendulumSharedModel=PendulumSharedModel,
        CurvaturePendulumModel=CurvaturePendulumModel,
        PostPendulumModel=PostPendulumModel,
        StatePostPendulumModel=StatePostPendulumModel,
        PrePostPendulumModel=PrePostPendulumModel,
        TimePostPendulumModel=TimePostPendulumModel,
        SpringMassModel=SpringMassModel,
        ThreeBodySpringMassModel=ThreeBodySpringMassModel,
        ThreeBodySpringMassGraphModel=ThreeBodySpringMassGraphModel,
        LennardJonesMLP=LennardJonesMLP
    )

    module = module_mapping[module]
    model = module(module_config)

    return model
