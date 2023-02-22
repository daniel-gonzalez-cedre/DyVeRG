from collections import Counter

import numpy as np
import networkx as nx


# scalar-valued statistics
def average_degree(g: nx.Graph) -> float:
    '''the average degree of g'''
    return np.mean([g.degree(v) for v in g])


def triangle_count(g: nx.Graph) -> int:
    '''the total number of triangles in g (ignoring self-edges)'''
    return sum(nx.triangles(g).values()) // 3


def clustering(g: nx.Graph) -> float:
    '''the average clustering coefficient of g'''
    return nx.average_clustering(g)


def transitivity(g: nx.Graph) -> float:
    '''the transitivity of g'''
    return nx.transitivity(g)


# vector-valued statistics
def degree_distribution(g: nx.Graph) -> list[int]:
    '''a distribution (dist[deg] = freq) giving the frequency of each degree in g'''
    degrees = Counter(dict(nx.degree(g)).values())
    return [degrees[deg] for deg in range(max(degrees) + 1)]


def spectrum(g: nx.Graph) -> np.ndarray[float]:
    '''the eigenvalue spectrum of the Laplacian of the (multi)graph g'''
    return nx.laplacian_spectrum(g)
