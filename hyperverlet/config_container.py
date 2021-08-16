preset_config_paths = {
    'integrator_comparison': [
        'configurations/integrator_comparison/{system}/euler.json',
        'configurations/integrator_comparison/{system}/hypereuler.json',
        'configurations/integrator_comparison/{system}/heun.json',
        'configurations/integrator_comparison/{system}/hyperheun.json',
        'configurations/integrator_comparison/{system}/velocityverlet.json',
        'configurations/integrator_comparison/{system}/hyperverlet.json',
        'configurations/integrator_comparison/{system}/ruth4.json',
        'configurations/integrator_comparison/{system}/rk4.json',
        'configurations/integrator_comparison/{system}/symplectichyperverlet.json',
        'configurations/integrator_comparison/{system}/symplectichyperverlet_extended.json',
        'configurations/integrator_comparison/{system}/sympnet.json',
        'configurations/integrator_comparison/{system}/sympnet_extended.json',
    ],
    'hyperruth': [
        'configurations/integrator_comparison/{system}/velocityverlet.json',
        'configurations/integrator_comparison/{system}/hyperverlet.json',
        'configurations/integrator_comparison/{system}/ruth4.json',
        'configurations/integrator_comparison/{system}/hyperruth.json',
    ],
    'hypersolver_placement_both': [
        'configurations/ablation/hypersolver_placement/{system}/hyperverlet.json',
        'configurations/ablation/hypersolver_placement/{system}/alternating_hyperverlet.json',
        'configurations/ablation/hypersolver_placement/{system}/sequentialpost_hyperverlet.json'
    ],
    'hypersolver_placement_single': [
        'configurations/ablation/hypersolver_placement/{system}/ponly_hyperverlet.json',
        'configurations/ablation/hypersolver_placement/{system}/qonly_hyperverlet.json',
    ],
    'shared_unshared': [
        'configurations/ablation/shared_unshared/{system}/hyperverlet_shared.json',
        'configurations/ablation/shared_unshared/{system}/hyperverlet_unshared.json'
    ],
    'model_input': [
        'configurations/ablation/model_input/{system}/hyperverlet_prepost.json',
        'configurations/ablation/model_input/{system}/hyperverlet_post.json',
        'configurations/ablation/model_input/{system}/hyperverlet_curvature.json',
        'configurations/ablation/model_input/{system}/hyperverlet_statepost.json',
        'configurations/ablation/model_input/{system}/hyperverlet_timepost.json'
    ],
    'loss_function': [
        'configurations/ablation/loss_function/{system}/hyperverlet_residual_l1.json',
        'configurations/ablation/loss_function/{system}/hyperverlet_residual_l2.json',
        'configurations/ablation/loss_function/{system}/hyperverlet_trajectory_l1.json',
        'configurations/ablation/loss_function/{system}/hyperverlet_trajectory_l2.json',
        'configurations/ablation/loss_function/{system}/hyperverlet_trajectory_timedecay.json'
    ],
    'generalization': [
        'configurations/generalization/out_of_distribution/pendulum_length/hyperverlet.json',
        'configurations/generalization/out_of_distribution/pendulum_mass/hyperverlet.json',
        'configurations/generalization/variable_parameters/pendulum_not_variable/hyperverlet.json',
        'configurations/generalization/variable_parameters/pendulum_variable/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_00/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_01/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_02/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_03/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_04/hyperverlet.json'
    ],
    'generalization_train_duration': [
        'configurations/generalization/train_duration/pendulum_00/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_01/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_02/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_03/hyperverlet.json',
        'configurations/generalization/train_duration/pendulum_04/hyperverlet.json'
    ],
    'generalization_variable_parameters': [
        'configurations/generalization/variable_parameters/pendulum_not_variable/hyperverlet.json',
        'configurations/generalization/variable_parameters/pendulum_variable/hyperverlet.json',
    ],
    'generalization_out_of_distribution': [
        'configurations/generalization/out_of_distribution/pendulum_mass/hyperverlet.json',
        'configurations/generalization/out_of_distribution/pendulum_length/hyperverlet.json',
    ],
    'total_energy': [
            'configurations/integrator_comparison/{system}/velocityverlet.json',
            'configurations/integrator_comparison/{system}/hyperheun.json',
            'configurations/integrator_comparison/{system}/hyperverlet.json',
            'configurations/integrator_comparison/{system}/ruth4.json',
            'configurations/integrator_comparison/{system}/rk4.json'
    ],
    'total_energy_full': [
            'configurations/integrator_comparison/{system}/velocityverlet.json',
            'configurations/integrator_comparison/{system}/hypereuler.json',
            'configurations/integrator_comparison/{system}/heun.json',
            'configurations/integrator_comparison/{system}/hyperheun.json',
            'configurations/integrator_comparison/{system}/hyperverlet.json',
            'configurations/integrator_comparison/{system}/ruth4.json',
            'configurations/integrator_comparison/{system}/rk4.json'
    ]
}