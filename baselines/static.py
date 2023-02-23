from functools import partial

import numpy as np
import networkx as nx

from src.decomposition import decompose
from dyverg.VRG import VRG
from dyverg.LightMultiGraph import LightMultiGraph as LMG


def VeRG(g: nx.Graph, time: int = None,
         mu: int = 4, clustering: int = 'leiden', verbose: bool = False) -> VRG:
    return decompose(g, time=(time if time else 0), mu=mu, clustering=clustering, verbose=verbose)


def uniform(g: nx.Graph, directed: bool = False) -> nx.Graph:
    return nx.gnm_random_graph(g.order(), g.size(), directed=directed)


def _uniform(g: nx.Graph, simple: bool = True):
    rng = np.random.default_rng()
    m = g.size()

    nodes = list(g.nodes())
    h = LMG() if isinstance(g, LMG) else nx.Graph()
    h.add_nodes_from(nodes)

    while m > 0:
        u, v = rng.choice(nodes, 2)
        if simple:
            while u == v:
                u, v = rng.choice(nodes, 2)

        if isinstance(h, LMG) or (u, v) not in h.edges():
            h.add_edge(u, v)
            m -= 1

    return h


def erdos_renyi(g: nx.Graph, directed: bool = False) -> nx.Graph:
    n = g.order()
    m = g.size()

    if not directed:
        p = (2 * m) / (n * (n - 1))
    else:
        p = m / (n * (n - 1))

    return nx.erdos_renyi_graph(n, p, directed=directed)


def _erdos_renyi(g: nx.Graph, directed: bool = False) -> nx.Graph:
    rng = np.random.default_rng()
    n = g.order()
    m = g.size()

    if not directed:
        p = (2 * m) / (n * (n - 1))
    else:
        p = m / (n * (n - 1))

    h = LMG() if isinstance(g, LMG) else nx.Graph()
    h.add_nodes_from(g.nodes())

    for u in h:
        for v in h:
            if rng.choice((True, False), p=(p, 1 - p)):
                h.add_edge(u, v)

    return h


def chung_lu(g: nx.Graph) -> nx.Graph:
    w = list(dict(nx.degree(g)).values())
    return nx.expected_degree_graph(w, selfloops=(nx.number_of_selfloops(g) > 0))


def _chung_lu(g: nx.Graph, simple: bool = True):
    rng = np.random.default_rng()
    degrees = {v: g.degree(v) for v in g}

    h = LMG() if isinstance(g, LMG) else nx.Graph()
    h.add_nodes_from(g)

    while degrees:
        nodes = list(degrees.keys())
        u, v = rng.choice(nodes, 2)

        if simple:
            while u == v:
                u, v = rng.choice(nodes, 2)

        assert degrees[u] != 0
        assert degrees[v] != 0

        if isinstance(h, LMG) or (u, v) not in h.edges():
            h.add_edge(u, v)

            if degrees[u] == 1:
                del degrees[u]
            else:
                degrees[u] -= 1

            if degrees[v] == 1:
                del degrees[v]
            else:
                degrees[v] -= 1

    return h


def _chung_lu_switch(g: nx.Graph, num_switches: int = None, simple: bool = True):
    rng = np.random.default_rng()
    num_switches = num_switches if num_switches else g.size()
    h = g.copy()

    while num_switches > 0:
        e, f = rng.choice(list(h.edges()), 2)
        eu, ev = e
        fu, fv = f

        if simple and ((eu, fv) in h.edges() or (fu, ev) in h.edges()):
            continue

        h.remove_edge(eu, ev)
        h.remove_edge(fu, fv)
        h.add_edge(eu, ev)
        h.add_edge(fu, fv)
        num_switches -= 1

    return h


def watts_strogatz(g: nx.Graph):
    rng = np.random.default_rng()
    n: int = g.order()
    m: int = g.size()
    avg: float = 2 * m / n

    k: int = int(avg)
    p: float = avg - k

    def ws_distance(u: int, v: int):
        return min((u - v) % n, (v - u) % n)

    h = LMG() if isinstance(g, LMG) else nx.Graph()
    h.add_nodes_from(list(range(n)))

    for u in h:
        k_ball = [v for v in h
                  if (u != v) and (ws_distance(u, v) <= k)]
        k_ball = sorted(k_ball, key=partial(ws_distance, u))

        for v in k_ball[:k]:
            if (u, v) not in h.edges():
                h.add_edge(u, v)

        if (p == 1.0) or (rng.uniform() < p):
            for v in k_ball[k:]:
                if (u, v) not in h.edges():
                    h.add_edge(u, v)
                    break

    return h


def barabasi_albert(g: nx.Graph):
    pass


def stochastic_block(g: nx.Graph):
    raise NotImplementedError
    # return nx.stochastic_block_model()


def cnrg(g: nx.Graph):
    raise NotImplementedError
