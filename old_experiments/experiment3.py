import sys
import pickle
import random
from typing import Dict

from tqdm import tqdm
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

sys.path.append('../../cnrg')
sys.path.append('../../cnrg/cnrg')

from data_scripts import read_data, create_graphs
from rule_to_rule_scripts import convert_LMG, decompose, ancestor, common_ancestor
from rule_to_rule_scripts import update_grammar_independent, update_rule_case1, update_rule_case2


def plotting():
    global DATANAME, PLLS, PS

    xs = np.asarray(sorted(PLLS.keys()))
    true_y = [PLLS[0] for x in xs]
    yavg = [np.mean(PLLS[x]) for x in xs]
    ymax = [np.max(PLLS[x]) for x in xs]
    ymin = [np.min(PLLS[x]) for x in xs]

    with plt.style.context(['ipynb', 'use_mathtext', 'colors5-light']):
        plt.title('Conditional LL under different levels of random perturbation')
        plt.xlabel('percentage (%) of total edges randomly rewired')
        plt.ylabel('log likelihood')
        plt.plot(xs * 100, true_y)
        plt.plot(xs * 100, yavg)
        plt.fill_between(xs * 100, ymax, ymin, alpha=0.2)
        plt.legend(['true LL', 'mean of perturbed LL\'s'])
        plt.ticklabel_format(style='plain')
        plt.savefig(f'../figures/rule_to_rule_{DATANAME}_exp1.svg')


def perturb_graph(g: nx.Graph, p_ratio: float = .01) -> nx.Graph:
    global GRAPHS, BASE_GRAMMAR, PGRAMMARS, PLLS

    h = g.copy()
    num_edges = int(h.size() * p_ratio)
    edge_sample = random.sample(h.edges(), num_edges)

    for old_u, old_v in edge_sample:
        h.remove_edge(old_u, old_v)
        u, v = random.sample(h.nodes(), 2)

        while (u, v) in h.edges():
            u, v = random.sample(h.nodes(), 2)

        h.add_edge(u, v)

    while not nx.is_connected(h):
        component1, component2 = random.sample(list(nx.connected_components(h)), 2)
        x1, = random.sample(component1, 1)
        x2, = random.sample(component2, 1)
        h.add_edge(x1, x2)

    return h


# def exp_helper(p, graph, base_grammar, base_graph):
def exp_helper(index: int, prob: float):
    global GRAPHS, BASE_GRAMMAR, PGRAMMARS

    base_grammar = BASE_GRAMMARS[index - 1]
    base_graph = GRAPHS[index - 1]
    cond_graph = GRAPHS[index]

    p_graph = perturb_graph(cond_graph, p_ratio=prob)
    p_grammar = update_grammar_independent(base_grammar, base_graph, p_graph)
    return p_grammar, index, prob


# datanames = ['nips', 'fb-messages']
if __name__ == '__main__':
    parallel = True
    redundancy = 5
    DATANAME = 'fb-messages'
    lookback = 10

    GRAPHS, _ = read_data(dataname=DATANAME, lookback=lookback)
    BASE_GRAMMARS = [decompose(graph) for graph in GRAPHS[:-1]]

    # for itr, GRAPH in enumerate(GRAPHS[1:]):

    ps = np.linspace(0.01, 0.2, 10)
    args = [(idx, p) for idx in range(1, len(GRAPHS)) for p in ps]

    PGRAMMARS = {idx: {p: [] for p in ps} for idx in range(1, len(GRAPHS))}  # typing: ignore
    PLLS = {idx: {p: [] for p in ps} for idx in range(1, len(GRAPHS))}  # typing: ignore

    if parallel:
        for itr in range(redundancy):
            print(f'starting iteration {itr}...', end=' ')
            results = Parallel(n_jobs=20)(delayed(exp_helper)(idx, p) for idx, p in args)

            for grammar, idx, p in results:
                PGRAMMARS[idx][p].append(grammar)
                PLLS[idx][p].append(grammar.conditional_ll())

            print('done')
    else:
        raise NotImplementedError
        # for itr in range(redundancy):
        #     print(f'starting iteration {itr}...', end=' ')
        #     for p in ps:
        #         pgraph = perturb_graph(GRAPHS[1], p_ratio=p)
        #         pgrammar = update_grammar_independent(BASE_GRAMMAR, GRAPHS[0], pgraph)
        #         PGRAMMARS[p] = pgrammar
        #         PLLS[p] = pgrammar.conditional_ll()

        #     print('done')

    for idx in range(1, len(GRAPHS)):
        true_grammar = update_grammar_independent(BASE_GRAMMARS[idx - 1], GRAPHS[idx - 1], GRAPHS[idx])
        PGRAMMARS[idx][0] = true_grammar
        PLLS[idx][0] = true_grammar.conditional_ll()

    # with open(f'../results/{DATANAME}_{lookback}_exp3.grammars', 'wb') as outfile:
    #     pickle.dump(PGRAMMARS, outfile)

    with open(f'../results/{DATANAME}_{lookback}_exp3.lls', 'wb') as outfile:
        pickle.dump(PLLS, outfile)
