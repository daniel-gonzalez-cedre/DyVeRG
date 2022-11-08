import os
import sys
import random
import pickle
from typing import List, Dict

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
    global DATANAME, SLLS

    xs = np.asarray(sorted(SLLS.keys()))
    # true_y = [SLLS[0] for x in xs]
    ys = [SLLS[x] for x in xs]
    # ymax = [np.max(SLLS[x]) for x in xs]
    # ymin = [np.min(SLLS[x]) for x in xs]

    with plt.style.context(['ipynb', 'use_mathtext', 'colors5-light']):
        plt.title('Conditional LL under different levels of random perturbation')
        plt.xlabel('percentage (%) of total edges randomly rewired')
        plt.ylabel('log likelihood')
        # plt.plot(xs * 100, true_y)
        plt.plot(xs, ys)
        # plt.fill_between(xs * 100, ymax, ymin, alpha=0.2)
        # plt.legend(['true LL', 'mean of perturbed LL\'s'])
        plt.ticklabel_format(style='plain')
        plt.savefig(f'../figures/rule_to_rule_{DATANAME}_exp2.svg')


def exp_helper(idx):
    global BASE_GRAMMAR, GRAPHS
    result = update_grammar_independent(BASE_GRAMMAR, GRAPHS[0], GRAPHS[idx])
    return result, idx


# datanames = ['nips', 'fb-messages']
if __name__ == '__main__':
    parallel = True
    DATANAME = 'fb-messages'
    lookback = 9

    GRAPHS, _ = read_data(dataname=DATANAME, lookback=lookback)
    BASE_GRAMMAR = decompose(GRAPHS[0])

    SGRAMMARS = {}
    SLLS = {}

    if parallel:
        results = Parallel(n_jobs=len(GRAPHS[2:]))(delayed(exp_helper)(idx) for idx in range(2, len(GRAPHS)))

        for grammar, index in results:
            SGRAMMARS[index] = grammar
            SLLS[index] = grammar.conditional_ll()
    else:
        pass

    true_grammar = update_grammar_independent(BASE_GRAMMAR, GRAPHS[0], GRAPHS[1])
    true_ll = true_grammar.conditional_ll()
    SGRAMMARS[1] = true_grammar
    SLLS[1] = true_grammar.conditional_ll()

    # with open(f'../results/{DATANAME}_{lookback}_exp2.grammars', 'wb') as outfile:
    #     pickle.dump(SGRAMMARS, outfile)

    with open(f'../results/{DATANAME}_{lookback}_exp2.lls', 'wb') as outfile:
        pickle.dump(SLLS, outfile)
