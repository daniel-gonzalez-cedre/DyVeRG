from tqdm import tqdm
import networkx as nx

from src.decomposition import decompose
from src.adjoin_graph import update_grammar
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


def GraphRNN(graphs: list[nx.Graph]):
    raise NotImplementedError
