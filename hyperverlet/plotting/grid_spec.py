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


def gs_4_3_2(fig):
    gs = GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[:-1, :1])
    ax2 = fig.add_subplot(gs[0, -1])
    ax3 = fig.add_subplot(gs[1, -1])
    ax4 = fig.add_subplot(gs[2, :])
    return ax1, ax2, ax3, ax4


def gs_5_3_3(fig):
    gs = GridSpec(3, 3)
    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[-1, 0])
    ax4 = fig.add_subplot(gs[-1, 1])
    ax5 = fig.add_subplot(gs[-1, 2])
    return ax1, ax2, ax3, ax4, ax5


def gs_5_3_2(fig):
    gs = GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[:2, :1])
    ax2 = fig.add_subplot(gs[-1, :1])
    ax3 = fig.add_subplot(gs[0, -1])
    ax4 = fig.add_subplot(gs[1, -1])
    ax5 = fig.add_subplot(gs[2, -1])
    return ax1, ax2, ax3, ax4, ax5


def gs_6_4_2(fig):
    gs = GridSpec(4, 2)
    ax1 = fig.add_subplot(gs[:2, :1])
    ax2 = fig.add_subplot(gs[-1, :1])
    ax3 = fig.add_subplot(gs[0, -1])
    ax4 = fig.add_subplot(gs[1, -1])
    ax5 = fig.add_subplot(gs[2, -1])
    ax6 = fig.add_subplot(gs[3, -1])
    return ax1, ax2, ax3, ax4, ax5, ax6


def gs_line(fig, ncol):
    gs = GridSpec(1, ncol)
    return [fig.add_subplot(gs[0, i]) for i in range(ncol)]
