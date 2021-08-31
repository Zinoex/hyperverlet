from hyperverlet.models.pendulum import PendulumModel, SymplecticPendulumModel
from hyperverlet.models.spring_mass import SpringMassModel, SymplecticSpringMassModel


def construct_model(module_config):
    module = module_config["module"]

    module_mapping = dict(
        PendulumModel=PendulumModel,
        SymplecticPendulumModel=SymplecticPendulumModel,
        SpringMassModel=SpringMassModel,
        SymplecticSpringMassModel=SymplecticSpringMassModel
    )

    module = module_mapping[module]
    model = module(module_config)

    return model
