from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.datasets import PendulumDataset, SpringMassDataset, ThreeBodySpringMassDataset, \
    ThreeBodyGravityDataset, DoublePendulumDataset, SymmetricSpringMassDataset
from hyperverlet.utils.misc import format_path


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

    train_cache_path = format_path(config, dataset_config["cache_path"], split='train')
    test_cache_path = format_path(config, dataset_config["cache_path"], split='test')

    coarsening_factor = dataset_config['coarsening_factor']
    duration_stddev = dataset_config['duration_stddev']

    test_experiment_args = dataset_config.get('test_experiment_args', {})
    train_experiment_args = dataset_config.get('train_experiment_args', {})

    gt_solver = construct_solver(gt_solver_name)

    ds_mapping = dict(pendulum=PendulumDataset, double_pendulum=DoublePendulumDataset, spring_mass=SpringMassDataset, symmetric_spring_mass=SymmetricSpringMassDataset, three_body_spring_mass=ThreeBodySpringMassDataset, three_body_gravity=ThreeBodyGravityDataset)
    ds_cls = ds_mapping[dataset]

    test_ds = ds_cls(gt_solver, test_duration, 0, test_trajectory_length, test_num_config, coarsening_factor, test_cache_path, sequence_length=None, **test_experiment_args)

    train_ds = None
    if trainable:
        train_ds = ds_cls(gt_solver, train_duration, duration_stddev, train_trajectory_length, train_num_config, coarsening_factor, train_cache_path, sequence_length=train_sequence_length, **train_experiment_args)

    return train_ds, test_ds
