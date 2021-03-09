import numpy as np


def calc_kinetic_energy(m, l, p):
    omega = p / m
    v = l * omega
    return 0.5 * m * v ** 2


def calc_potential_energy(m, g, l, q):
    return m * g * l * (1 - np.cos(q))


def calc_total_energy(kin, pot):
    return kin + pot
