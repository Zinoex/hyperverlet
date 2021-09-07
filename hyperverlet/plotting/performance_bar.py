import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from hyperverlet.plotting.utils import fetch_result_dict
from hyperverlet.utils.measures import z_loss


def performance_bar(config):
    performance_bar_config = config['performance_bar_plot']

    mse = []
    labels = []

    for idx, (label, path) in enumerate(config['results'].items()):
        result_dict = fetch_result_dict(path)

        q = result_dict["q"]
        p = result_dict["p"]
        gt_q = result_dict['gt_q']
        gt_p = result_dict['gt_p']

        z_mse = z_loss(q, p, gt_q, gt_p).numpy()
        label = np.full((z_mse.shape[0],), label)

        mse.append(z_mse)
        labels.append(label)

    mse = np.concatenate(mse).astype(object)
    labels = np.concatenate(labels).astype(object)
    data = np.stack([labels, mse], axis=1)

    df = pd.DataFrame(data=data, columns=['Solver', 'MSE'])

    sns.barplot(x='Solver', y='MSE', data=df, ci=95, capsize=0.2)
    plt.xticks(rotation=-30, horizontalalignment='center')

    plot_path = performance_bar_config['plot_path']
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved at {plot_path}")
