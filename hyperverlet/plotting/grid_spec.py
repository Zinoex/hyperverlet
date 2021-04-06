from matplotlib.gridspec import GridSpec


def gs_3_2_3(fig):
    gs = GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])
    return ax1, ax2, ax3


def gs_2_1_2(fig):
    gs = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    return ax1, ax2