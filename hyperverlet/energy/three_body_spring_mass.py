import numpy as np


def calc_kinetic_energy(m, p):
    v = p / m
    return (0.5 * m * v ** 2).reshape(p.shape[0], -1).sum(axis=1)


def displacement(q):
    a = np.expand_dims(q, -2)
    b = np.expand_dims(q, -3)
    return a - b


def distance(disp):
    return np.linalg.norm(disp, axis=-1)


def calc_potential_energy(k, q, l):
    disp = displacement(q)
    dist = distance(disp)
    x = dist - l

    return (0.5 * k * x ** 2).reshape(q.shape[0], -1).sum(axis=1)


def calc_total_energy(kin, pot):
    return kin + pot
