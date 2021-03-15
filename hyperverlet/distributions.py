import numpy as np
import torch


def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    one = 1
    epsilon = torch.finfo(p.dtype).eps
    v = torch.clamp(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * torch.sqrt(torch.tensor([2.0])) * torch.erfinv(v)
    x = torch.clamp(x, a, b)

    return x


def sample_parameterized_truncated_normal(shape, mu, sigma, a, b):
    return parameterized_truncated_normal(torch.rand(shape), mu, sigma, a, b)
