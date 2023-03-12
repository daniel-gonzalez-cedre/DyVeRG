from collections import Counter

import numpy as np
import networkx as nx


# scalar-valued statistics
def average_degree(g: nx.Graph) -> float:
    '''the average degree of g'''
    if g.order() == 0:
        return 0
    return [np.mean([g.degree(v) for v in g])]


def triangle_count(g: nx.Graph) -> int:
    '''the total number of triangles in g (ignoring self-edges)'''
    if g.order() == 0:
        return 0
    return sum(nx.triangles(g).values()) // 3


def clustering(g: nx.Graph) -> float:
    '''the average clustering coefficient of g'''
    if g.order() == 0:
        return 0
    return nx.average_clustering(g)


def transitivity(g: nx.Graph) -> float:
    '''the transitivity of g'''
    if g.order() == 0:
        return 0
    return nx.transitivity(g)


# vector-valued statistics
def degree_distribution(g: nx.Graph) -> list[int]:
    '''a distribution (dist[deg] = freq) giving the frequency of each degree in g'''
    if g.order() == 0:
        return [0]
    degrees = Counter(dict(nx.degree(g)).values())
    return [degrees[deg] for deg in range(max(degrees) + 1)]


def spectrum(g: nx.Graph) -> np.ndarray[float]:
    '''the eigenvalue spectrum of the Laplacian of the (multi)graph g'''
    if g.order() == 0:
        return [0]
    return nx.laplacian_spectrum(g)
