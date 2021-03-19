from hyperverlet.models.lennard_jones import LennardJonesMLP
from hyperverlet.models.pendulum import PendulumModel
from hyperverlet.models.spring_mass import SpringMassModel
from hyperverlet.models.three_body_spring_mass import ThreeBodySpringMassModel
from hyperverlet.utils import load_config


def construct_model(model_path):
    model_config = load_config(model_path)
    module = model_config['module']

    module_mapping = dict(
        PendulumModel=PendulumModel,
        SpringMassModel=SpringMassModel,
        ThreeBodySpringMassModel=ThreeBodySpringMassModel,
        LennardJonesMLP=LennardJonesMLP
    )

    module = module_mapping[module]
    model = module(model_config)

    return model
