import torch


def wrap_pbc(x, bbox):
    x += (x < -bbox / 2) * bbox
    x -= (x > bbox / 2) * bbox

    return x


def wrap_pbc_torch(x, bbox, repeat=1):
    bbox = bbox.view(*[1 for _ in range(len(x.size()) - 1)], -1)

    for _ in range(repeat):
        x += (x < -bbox / 2) * bbox
        x -= (x > bbox / 2) * bbox

    return x
