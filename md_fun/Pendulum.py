import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


class Pendulum:
    '''
    This object - MrDynamics will generate list of co-ordinate and velocity of a
    particle.

    g is the gravitational accelaration
    l is the length of an arm
    m is a mass of the particle.

    Tmax is the simulation time.
    Δt is the time step which act as a accuracy of Borish scheme.

    θ and ω are the initial angle and angular velocity, respectively.

    '''

    def __init__(self, g, l, m, Tmax, delta_t, e, w):

        self.l = l

        # list of position, velocity, and time
        self.q = [e]
        self.p = [w]
        self.time = []

        # accelaration
        self.Acc = lambda e, g, l: -(g / l) * np.sin(e)

        # Run the algorithm
        self.Beeman(g, l, Tmax, delta_t, e, w)

        # Energy
        self.KineticEnergy = 0.5 * m * (l * np.array(self.p)) ** 2
        self.PotentialEnergy = -m * g * l * np.cos(np.array(self.q))
        self.TotalEnergy = self.KineticEnergy + self.PotentialEnergy

    def Beeman(self, g, l, Tmax, delta_t, e, w):

        t = 0

        # Step - 1 : Initialization
        ao = self.Acc(e, g, l)

        w += ao * delta_t
        e += w * delta_t
        t += delta_t

        a1 = self.Acc(e, g, l)

        self.p.append(w)
        self.q.append(e)

        # Step - 2 : Beeman Algorithm
        while (t <= Tmax):
            e += w * delta_t + (4 * a1 - ao) * delta_t ** 2 / 6

            a2 = self.Acc(e, g, l)

            w += (5 * a2 + 8 * a1 - ao) * delta_t / 12

            self.p.append(w)
            self.q.append(e)

            ao = a1
            a1 = a2
            t += delta_t

        self.time = np.linspace(0, t, len(self.q))

    def PlotTrajectory(self):

        fig = plt.figure(figsize=(80, 60))
        gs = GridSpec(1, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # PLOT - 2: Energy
        ax2.scatter(self.time[0], self.TotalEnergy[0], label=r'$E_{sys}$')
        ax2.scatter(self.time[0], self.KineticEnergy[0], label=r'KE')
        ax2.scatter(self.time[0], self.PotentialEnergy[0], label=r'PE')
        ax2.legend()

        for i in range(1, len(self.q)):
            # PLOT - 1: Model
            ax1.clear()

            x, y = self.l * np.sin(self.q[i]), - self.l * np.cos(self.q[i])

            ax1.plot([0, x], [0, y], linewidth=3)
            ax1.scatter(x, y, color='red', marker='o',
                        s=500, alpha=0.8)
            ax1.set_xlabel('X', fontweight='bold', fontsize=14)
            ax1.set_ylabel('Y', fontweight='bold', fontsize=14)
            ax1.set_xlim([-self.l - 0.5, self.l + 0.5])
            ax1.set_ylim([-self.l - 0.5, self.l + 0.5])

            # PLOT - 2: Energy
            ax2.plot([self.time[i - 1], self.time[i]], [self.TotalEnergy[i - 1], self.TotalEnergy[i]], color='blue')
            ax2.plot([self.time[i - 1], self.time[i]], [self.KineticEnergy[i - 1], self.KineticEnergy[i]],
                     color='orange')
            ax2.plot([self.time[i - 1], self.time[i]], [self.PotentialEnergy[i - 1], self.PotentialEnergy[i]],
                     color='green')

            plt.pause(1E-11)