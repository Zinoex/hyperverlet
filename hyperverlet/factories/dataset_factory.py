from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.experiments import PendulumDataset, SpringMassDataset, ThreeBodySpringMassDataset


def construct_dataset(config):
    dataset_config = config["dataset_args"]
    dataset = dataset_config['dataset']

    gt_solver_name = dataset_config['gt_solver']

    train_duration = dataset_config['train_duration']
    test_duration = dataset_config['test_duration']

    train_trajectory_length = dataset_config['train_trajectory_length']
    test_trajectory_length = dataset_config['test_trajectory_length']

    train_sequence_length = dataset_config["train_sequence_length"]

    num_config = dataset_config['num_configurations']
    coarsening_factor = dataset_config['coarsening_factor']

    gt_solver = construct_solver(gt_solver_name)

    if dataset == 'pendulum':
        train_ds = PendulumDataset(gt_solver, train_duration, train_trajectory_length, num_config, coarsening_factor, sequence_length=train_sequence_length)
        test_ds = PendulumDataset(gt_solver, test_duration, test_trajectory_length, 1, coarsening_factor, sequence_length=None)
    elif dataset == 'spring_mass':
        train_ds = SpringMassDataset(gt_solver, train_duration, train_trajectory_length, num_config, coarsening_factor, sequence_length=train_sequence_length)
        test_ds = SpringMassDataset(gt_solver, test_duration, test_trajectory_length, 1, coarsening_factor, sequence_length=None)
    elif dataset == 'three_body_spring_mass':
        train_ds = ThreeBodySpringMassDataset(gt_solver, train_duration, train_trajectory_length, num_config, coarsening_factor, sequence_length=train_sequence_length)
        test_ds = ThreeBodySpringMassDataset(gt_solver, test_duration, test_trajectory_length, 1, coarsening_factor, sequence_length=None)
    else:
        raise NotImplementedError

    return train_ds, test_ds
