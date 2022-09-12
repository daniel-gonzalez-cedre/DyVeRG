import sys
import pickle
import random
from typing import Dict

from tqdm import tqdm
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

from data_scripts import read_data, create_graphs
from rule_to_rule_scripts import convert_LMG, decompose, ancestor, common_ancestor
from rule_to_rule_scripts import update_grammar_independent, update_rule_case1, update_rule_case2

sys.path.append('../../cnrg')
sys.path.append('../../cnrg/cnrg')


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
def exp_helper(prob: float):
    global GRAPHS, BASE_GRAMMAR, PGRAMMARS, PLLS

    pgraph = perturb_graph(GRAPHS[1], p_ratio=prob)
    pgrammar = update_grammar_independent(BASE_GRAMMAR, GRAPHS[0], pgraph)
    return pgrammar, prob


# datanames = ['nips', 'fb-messages', 'ca-cit-HepPh', 'tech-as-topology']
if __name__ == '__main__':
    parallel = True
    redundancy = 5
    prop = 0.2
    DATANAME = 'fb-messages'
    lookback = 9

    GRAPHS, _ = read_data(dataname=DATANAME, lookback=lookback)
    BASE_GRAMMAR = decompose(GRAPHS[0])

    # for rule in tqdm(BASE_GRAMMAR.rule_list, desc='COMPUTING CANON MATRICES'):
    #     rule.compute_canon_matrix()

    PS = np.linspace(0.001, prop, 10)

    PGRAMMARS = {p: [] for p in PS}  # typing: ignore
    PLLS = {p: [] for p in PS}  # typing: ignore

    if parallel:
        for itr in range(redundancy):
            print(f'starting iteration {itr}...', end=' ')
            results = Parallel(n_jobs=10)(delayed(exp_helper)(p) for p in PS)

            for grammar, key in results:
                PGRAMMARS[key].append(grammar)
                PLLS[key].append(grammar.conditional_ll())

            print('done')
    else:
        for itr in range(redundancy):
            print(f'starting iteration {itr}...', end=' ')
            for p in PS:
                pgraph = perturb_graph(GRAPHS[1], p_ratio=p)
                pgrammar = update_grammar_independent(BASE_GRAMMAR, GRAPHS[0], pgraph)
                PGRAMMARS[p] = pgrammar
                PLLS[p] = pgrammar.conditional_ll()

            print('done')

    true_grammar = update_grammar_independent(BASE_GRAMMAR, GRAPHS[0], GRAPHS[1])
    true_ll = true_grammar.conditional_ll()
    PGRAMMARS[0] = true_grammar
    PLLS[0] = true_ll

    # with open(f'../results/{DATANAME}_{lookback}_exp1_{int(100 * prop)}.grammars', 'wb') as outfile:
    #     pickle.dump(PGRAMMARS, outfile)

    with open(f'../results/{DATANAME}_{lookback}_exp1_{int(100 * prop)}.lls', 'wb') as outfile:
        pickle.dump(PLLS, outfile)
