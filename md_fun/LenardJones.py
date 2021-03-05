import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from solvers import Beeman, Verlet


class LenardJones:
    def __init__(self, eps, rm, m, Tmax, delta_t, r, v, solver='Beeman'):

        self.eps, self.rm = eps, rm

        # accelaration
        acc_func = lambda r, eps, rm, m: (12 * eps / rm ** 2) * ((rm / r) ** 14 - (rm / r) ** 8) * r / m

        # Run the algorithm
        if solver == 'Beeman':
            res_r, res_v, t = Beeman(acc_func, Tmax, delta_t, r, v, eps, rm, m)
        elif solver == 'Verlet':
            res_r, res_v, t = Verlet(acc_func, Tmax, delta_t, r, v, eps, rm, m)

        self.res_r = res_r
        self.res_v = res_v
        self.time = t

        # Energy
        self.KE = 0.5 * m * np.array(self.res_v) ** 2
        self.PE = np.array([eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6) for r in self.res_r])
        self.E = self.KE + self.PE

    def PlotTrajectory(self):
        fig = plt.figure(figsize=(80, 60))
        gs = GridSpec(1, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # PLOT - 2: Energy
        ax2.scatter(self.time[0], self.E[0], label=r'$E_{sys}$')
        ax2.scatter(self.time[0], self.KE[0], label=r'KE')
        ax2.scatter(self.time[0], self.PE[0], label=r'PE')
        ax2.legend()

        for i in range(1, len(self.time)):
            # PLOT - 1: Model
            ax1.clear()

            ax1.scatter(self.res_r[i], 0, color='blue', marker='o',
                        s=500, alpha=0.8)
            ax1.set_xlim([0, 3])
            ax1.set_xlabel('X', fontweight='bold', fontsize=14)
            ax1.set_ylabel('Y', fontweight='bold', fontsize=14)
            ax1.grid()

            # Potential energy curve
            r = np.linspace(0.8, 3, 100)
            U = np.array(self.eps * ((self.rm / r) ** 12 - 2 * (self.rm / r) ** 6))
            ax1.plot(r, U, 'r', label='LJ potential')
            ax1.legend()

            # PLOT - 2: Energy
            ax2.plot([self.time[i - 1], self.time[i]], [self.E[i - 1], self.E[i]], color='blue')
            ax2.plot([self.time[i - 1], self.time[i]], [self.KE[i - 1], self.KE[i]], color='orange')
            ax2.plot([self.time[i - 1], self.time[i]], [self.PE[i - 1], self.PE[i]], color='green')

            plt.pause(1E-11)