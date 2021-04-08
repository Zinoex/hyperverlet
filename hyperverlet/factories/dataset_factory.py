from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.datasets import PendulumDataset, SpringMassDataset, ThreeBodySpringMassDataset


def construct_dataset(config, trainable=True):
    dataset_config = config["dataset_args"]
    dataset = dataset_config['dataset']

    gt_solver_name = dataset_config['gt_solver']

    train_duration = dataset_config['train_duration']
    test_duration = dataset_config['test_duration']

    train_trajectory_length = dataset_config['train_trajectory_length']
    test_trajectory_length = dataset_config['test_trajectory_length']

    train_num_config = dataset_config['train_num_configurations']
    test_num_config = dataset_config['test_num_configurations']

    train_sequence_length = dataset_config["train_sequence_length"]

    coarsening_factor = dataset_config['coarsening_factor']

    gt_solver = construct_solver(gt_solver_name)

    ds_mapping = dict(pendulum=PendulumDataset, spring_mass=SpringMassDataset, three_body_spring_mass=ThreeBodySpringMassDataset)
    ds_cls = ds_mapping[dataset]

    test_ds = ds_cls(gt_solver, test_duration, test_trajectory_length, test_num_config, coarsening_factor, sequence_length=None)

    train_ds = None
    if trainable:
        train_ds = ds_cls(gt_solver, train_duration, train_trajectory_length, train_num_config, coarsening_factor, sequence_length=train_sequence_length)

    return train_ds, test_ds
