from functools import partial

import numpy as np
import networkx as nx
from tqdm import tqdm


def DyVeRG(graphs: list, times: list = None, mu: int = 4, clustering: int = 'leiden', verbose: bool = False):
    from src.decomposition import decompose
    from src.adjoin_graph import update_grammar
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


def VeRG(g: nx.Graph, t: int = None, mu: int = 4, clustering: int = 'leiden', verbose: bool = False):
    from src.decomposition import decompose
    return decompose(g, time=(t if t else 0), mu=mu, clustering=clustering, verbose=verbose)


def CNRG(g: nx.Graph):
    raise NotImplementedError


def graphRNN(graphs: list, nn: str = 'rnn') -> tuple:
    from baselines.graphrnn.fit import fit
    return fit(graphs, nn=nn)
    # return {'args': args, 'model': model, 'output': output}


def uniform(g: nx.Graph, directed: bool = False) -> nx.Graph:
    return nx.gnm_random_graph(g.order(), g.size(), directed=directed)


def erdos_renyi(g: nx.Graph, directed: bool = False) -> tuple:
    n = g.order()
    m = g.size()

    if not directed:
        p = (2 * m) / (n * (n - 1))
    else:
        p = m / (n * (n - 1))

    return n, p, directed
    # return nx.erdos_renyi_graph(n, p, directed=directed)


def chung_lu(g: nx.Graph) -> tuple:
    return list(dict(nx.degree(g)).values()), (nx.number_of_selfloops(g) > 0)
    # return nx.expected_degree_graph(w, selfloops=(nx.number_of_selfloops(g) > 0))


def stochastic_blockmodel(g: nx.Graph):
    from pyintergraph import nx2gt
    from graph_tool.all import minimize_blockmodel_dl as opt

    graph_gt = nx2gt(g)
    state = opt(graph_gt)
    return state
