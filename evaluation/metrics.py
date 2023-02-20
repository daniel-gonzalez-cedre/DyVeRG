import numpy as np
from networkx import Graph

from evaluation.mmd import mmd_rbf, mmd_poly, mmd_linear
from evaluation.portrait_divergence.portrait_divergence import portrait_divergence as pd


def maximum_mean_discrepancy(P: list[list], Q: list[list], kernel: str = 'gaussian', gamma: float = 1.0) -> float:
    '''
        P: list of samples [[...], [...], ... [...]]
        Q: list of samples [[...], [...], ... [...]]
    '''
    assert kernel in ('rbf', 'radial', 'gaussian', 'polynomial', 'linear')

    maxlenP = max(len(p) for p in P)
    maxlenQ = max(len(q) for q in Q)
    padlength = max(maxlenP, maxlenQ)

    for sample in (P + Q):
        sample += [0 for _ in range(padlength - len(sample))]

    X = np.asarray(P)
    Y = np.asarray(Q)

    if kernel in ('rbf', 'radial', 'gaussian'):
        value = mmd_rbf(X, Y, gamma=gamma)
    elif kernel == 'polynomial':
        value = mmd_poly(X, Y)
    else:
        value = mmd_linear(X, Y)

    return value


def portrait_divergence(g: Graph, h: Graph) -> float:
    return pd(g, h)
