import numpy as np


def calc_theta(p1, p2):
    vec = p2 - p1

    return np.arctan2(vec[1], vec[0]) + np.pi / 2


def calc_dist_2d(p1, p2):
    return np.linalg.norm(p1 - p2)