import numpy as np


def calc_kinetic_energy(m, p):
    v = p / m
    return 0.5 * m * v ** 2


def calc_potential_energy(k, q, l):
    x = q - l
    return 0.5 * k * x ** 2


def calc_total_energy(kin, pot):
    return kin + pot
