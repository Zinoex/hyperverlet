import numpy as np

def Verlet(acc_func, Tmax, delta_t, x, v, *args):
    t = 0

    # Step - 1 : Initialization for verlet
    x_0 = x

    x_res = [x]
    v_res = [v]
    time = []

    v += acc_func(x, *args) * delta_t
    x_1 = x_0 + v * delta_t
    t += delta_t

    x_res.append(x_1)

    # Step - 2 : Verlet Algorithm
    while (t <= Tmax):
        x2 = 2 * x_1 - x_0 + acc_func(x_1, *args) * delta_t ** 2

        x_res.append(x2)

        x_0 = x_1
        x_1 = x2
        t += delta_t

    # Step - 3 : Velocity
    Vel = lambda Pos, delta_t: [(Pos[i + 2] - Pos[i]) / (2 * delta_t) for i in range(0, len(Pos) - 2)]

    v_res = v_res + Vel(x_res, delta_t)
    x_res = x_res[:-1]
    time = np.linspace(0, t, len(x_res))

    return x_res, v_res, time


def Beeman(acc_func, Tmax, delta_t, x, v, *args):
    t = 0

    x_res = [x]
    v_res = [v]
    time = []

    # Step - 1 : Initialization for Beeman
    a_0 = acc_func(x, *args)

    v += a_0 * delta_t
    x += v * delta_t
    t += delta_t

    a1 = acc_func(x, *args)

    v_res.append(v)
    x_res.append(x)

    # Step - 2 : Beeman Algorithm
    while (t <= Tmax):
        x += v * delta_t + (4 * a1 - a_0) * delta_t ** 2 / 6

        a2 = acc_func(x, *args)

        v += (5 * a2 + 8 * a1 - a_0) * delta_t / 12

        v_res.append(v)
        x_res.append(x)

        a_0 = a1
        a1 = a2
        t += delta_t

    time = np.linspace(0, t, len(x_res))
    return x_res, v_res, time