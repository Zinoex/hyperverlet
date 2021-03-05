import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.gridspec import GridSpec


# %%
from solvers import Beeman, Verlet


class MassSpringSystem:
    '''
    This object - MrDynamics will generate list of co-ordinate and velocity of a
    particle.

    m is a mass of the particle.

    Tmax is the simulation time.
    Δt is the time step which act as a accuracy of Borish scheme.

    r and V are the initial position and velocity, respectively.

    '''

    def __init__(self, m, k, Tmax, delta_t, x, V):
        # accelaration
        calc_acceleration = lambda x, k, m: -(k / m) * x

        # numerical approch : Beeman
        x_res, v_res, time = Verlet(calc_acceleration, Tmax, delta_t, x, V, m, k)
        self.x_res = x_res
        self.v_res = v_res
        self.time = time

        # analytical result
        omg = (k / m) ** 0.5
        self.gt_position = x * np.cos(omg * self.time)
        self.gt_velcocity = -x * omg * np.sin(omg * self.time)

        # Error
        delta_x = 100 * (np.array(self.gt_position[1:]) - np.array(self.x_res[1:])) / np.array(self.gt_position[1:])
        delta_V = 100 * (np.array(self.gt_velcocity[1:]) - np.array(self.v_res[1:])) / np.array(self.gt_velcocity[1:])

        print('Median error in position ' + str(np.median(abs(delta_x))) + ' %')
        print('Median error in velocity ' + str(np.median(abs(delta_V))) + ' %')

    def PlotTrajectory(self):
        fig = plt.figure(figsize=(80, 60))
        gs = GridSpec(2, 1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])

        ax1.plot(self.time, self.gt_position, label='Analytical')
        ax1.plot(self.time, self.x_res, 'ro', alpha=0.5, label='Beeman')
        ax1.set_ylabel('x(t)', fontweight='bold', fontsize=14)
        ax1.grid()
        ax1.legend()

        ax2.plot(self.time, self.gt_velcocity)
        ax2.plot(self.time, self.v_res, 'ro', alpha=0.5)
        ax2.set_xlabel('time', fontweight='bold', fontsize=14)
        ax2.set_ylabel('v(t)', fontweight='bold', fontsize=14)
        ax2.grid()

        plt.show()