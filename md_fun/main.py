import numpy as np

from md_fun.LenardJones import LenardJones
from md_fun.MassSpringSystem import MassSpringSystem
from md_fun.Pendulum import Pendulum

if __name__ == '__main__':
    program = "pendulum"

    if program == 'lj':
        #Lenard Jones
        eps, rm, m = 2, 1, 1
        Tmax, delta_t = 8E-1, 5E-3
        r, v = 1.5, -2

        Answer = LenardJones(eps, rm, m, Tmax, delta_t, r, v, 'Verlet')
    elif program == 'mss':
        # Mass and Spring constant
        m, k = 1, 1
        Tmax, delta_t = 100, 0.2
        x, V = 1, 0

        Answer = MassSpringSystem(m, k, Tmax, delta_t, x, V)
    elif program == 'pendulum':
        g, l, m = 10, 1, 1
        Tmax, delta_t = 2, 1E-2
        e, w = np.pi / 4, 0

        Answer = Pendulum(g, l, m, Tmax, delta_t, e, w)
        pass


    Answer.PlotTrajectory()