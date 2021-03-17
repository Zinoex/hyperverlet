from hyperverlet.factories.solver_factory import construct_solver
from hyperverlet.experiments import PendulumDataset, SpringMassDataset, ThreeBodySpringMassDataset


def construct_dataset(config_args):
    dataset = config_args['dataset']
    gt_solver_name = config_args['gt_solver']
    duration = config_args['duration']
    traj_len = config_args['trajectory_length']
    num_config = config_args['num_configurations']
    coarsening_factor = config_args['coarsening_factor']

    gt_solver = construct_solver(gt_solver_name)
    if dataset == 'pendulum':
        train_ds = PendulumDataset(gt_solver, duration, traj_len, num_config, coarsening_factor, sequence_length=50)
        test_ds = PendulumDataset(gt_solver, duration, traj_len, 1, coarsening_factor, sequence_length=None)
    elif dataset == 'spring_mass':
        train_ds = SpringMassDataset(gt_solver, duration, traj_len, num_config, coarsening_factor, sequence_length=50)
        test_ds = SpringMassDataset(gt_solver, duration, traj_len, 1, coarsening_factor, sequence_length=None)
    elif dataset == 'three_body_spring_mass':
        train_ds = ThreeBodySpringMassDataset(gt_solver, duration, traj_len, num_config, coarsening_factor, sequence_length=50)
        test_ds = ThreeBodySpringMassDataset(gt_solver, duration, traj_len, 1, coarsening_factor, sequence_length=None)
    else:
        raise NotImplementedError

    return train_ds, test_ds
