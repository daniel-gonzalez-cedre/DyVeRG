from functools import partial

import numpy as np
import networkx as nx
from tqdm import tqdm

from baselines.graphrnn.fit import fit
# from baselines.graphrnn.train import *  # graphrnn
# from baselines.graphrnn.train import random, shuffle

from src.decomposition import decompose
from src.adjoin_graph import update_grammar
from dyverg.LightMultiGraph import LightMultiGraph as LMG
from dyverg.VRG import VRG


def DyVeRG(graphs: list[nx.Graphs], times: list[int] = None,
           mu: int = 4, clustering: int = 'leiden', verbose: bool = False) -> VRG:
    if not times:
        times = list(range(len(graphs)))

    prior_g = graphs[0]
    prior_t = times[0]
    grammar = decompose(prior_g, time=prior_t, mu=mu, clustering=clustering)

    for g, t in tqdm(zip(graphs[1:], times[1:]), desc='updating grammar', disable=(not verbose)):
        grammar = update_grammar(grammar, prior_g, g, prior_t, t)
        prior_g = g
        prior_t = t

    return grammar


def VeRG(g: nx.Graph, t: int = None, mu: int = 4, clustering: int = 'leiden', verbose: bool = False) -> VRG:
    return decompose(g, time=(t if t else 0), mu=mu, clustering=clustering, verbose=verbose)


def CNRG(g: nx.Graph):
    raise NotImplementedError


def graphRNN(graphs: list[nx.Graph], nn: str = 'rnn') -> tuple:
    return fit(graphs, nn=nn)
    # return {'args': args, 'model': model, 'output': output}


def uniform(g: nx.Graph, directed: bool = False) -> nx.Graph:
    return nx.gnm_random_graph(g.order(), g.size(), directed=directed)


def erdos_renyi(g: nx.Graph, directed: bool = False) -> nx.Graph:
    n = g.order()
    m = g.size()

    if not directed:
        p = (2 * m) / (n * (n - 1))
    else:
        p = m / (n * (n - 1))

    return nx.erdos_renyi_graph(n, p, directed=directed)


def chung_lu(g: nx.Graph) -> nx.Graph:
    w = list(dict(nx.degree(g)).values())
    return nx.expected_degree_graph(w, selfloops=(nx.number_of_selfloops(g) > 0))


def stochastic_blockmodel(g: nx.Graph):
    from pyintergraph import nx2gt
    from graph_tool.all import minimize_blockmodel_dl as opt

    graph_gt = nx2gt(g)
    state = opt(graph_gt)
    return state


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
