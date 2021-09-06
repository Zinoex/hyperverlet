import os

import seaborn as sns
from matplotlib import pyplot as plt

from hyperverlet.plotting.utils import fetch_result_dict


def canonical_plot(config):
    cm = sns.color_palette("muted")
    linestyles = [
         ('solid', 'solid'),
         ('dashed', 'dashed'),
         ('dotted', 'dotted'),
         ('dashdot', 'dashdot'),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
    ]

    canonical_config = config['canonical_plot']
    cfg = canonical_config['cfg_idx']
    spatial_size = canonical_config['spatial_size']
    start = canonical_config['start']
    end = canonical_config['end']
    fig, axs = plt.subplots(spatial_size * 2, sharex=True)

    for idx, (label, path) in enumerate(config['results'].items()):
        result_dict = fetch_result_dict(path)

        trajectory = result_dict["trajectory"][start:end, cfg]

        if canonical_config['include_ground_truth'] and idx == 0:
            q = result_dict['gt_q'][start:end, cfg]
            p = result_dict['gt_p'][start:end, cfg]

            kwargs = dict(label='Ground truth', linewidth=1.0, color=cm[0], linestyle=linestyles[0][1])

            for axis_idx in range(spatial_size):
                axs[axis_idx].plot(trajectory, q[..., axis_idx], **kwargs)
                axs[axis_idx + spatial_size].plot(trajectory, p[..., axis_idx], **kwargs)

        q = result_dict["q"][start:end, cfg]
        p = result_dict["p"][start:end, cfg]

        offset = 1 if canonical_config['include_ground_truth'] else 0
        kwargs = dict(label=label, linewidth=1.0, color=cm[idx + offset], linestyle=linestyles[idx + offset][1])

        for axis_idx in range(spatial_size):
            axs[axis_idx].plot(trajectory, q[..., axis_idx], **kwargs)
            axs[axis_idx + spatial_size].plot(trajectory, p[..., axis_idx], **kwargs)

    for axis_idx in range(spatial_size):
        axs[axis_idx].set_xlabel('Time')
        ylabel = f'q{axis_idx}' if spatial_size > 1 else 'q'
        axs[axis_idx].set_ylabel(ylabel)

        axs[axis_idx + spatial_size].set_xlabel('Time')
        ylabel = f'p{axis_idx}' if spatial_size > 1 else 'p'
        axs[axis_idx + spatial_size].set_ylabel(ylabel)
    axs[0].legend(loc='lower right')

    plot_path = canonical_config['plot_path']
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved at {plot_path}")
