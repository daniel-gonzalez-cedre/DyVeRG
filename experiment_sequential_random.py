import random
import pickle
from os import getcwd
from os.path import join
from argparse import ArgumentParser

import sys
sys.path.append('./src/')

import git
import numpy as np
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed

from cnrg.VRG import VRG
from src.utils import mkdir, silence
from src.data import load_data
from src.bookkeeping import decompose
from src.update_grammar import update_grammar


# g: the graph to perturb
# p: the percentage of the edges to rewire
def perturb_graph(g: nx.Graph, p: float = .01) -> nx.Graph:
    h = g.copy()

    if p == 0:
        return h

    num_edges = int(h.size() * p)
    edge_sample = random.sample(h.edges(), num_edges)

    for old_u, old_v in edge_sample:
        h.remove_edge(old_u, old_v)
        u, v = random.sample(h.nodes(), 2)

        while (u, v) in h.edges():
            u, v = random.sample(h.nodes(), 2)

        h.add_edge(u, v)

    # we no longer need to enforce connectivity on the graph
    # while not nx.is_connected(h):
    #     component1, component2 = random.sample(list(nx.connected_components(h)), 2)
    #     x1, = random.sample(component1, 1)
    #     x2, = random.sample(component2, 1)
    #     h.add_edge(x1, x2)

    return h


def experiment(curr_time: int, curr_graph: nx.Graph,
               next_time: int, next_graph: nx.Graph,
               p: float, mu: int):
    base_grammar = decompose(curr_graph, time=curr_time, mu=mu)

    perturbed_graph = perturb_graph(next_graph, p)

    joint_grammar = update_grammar(base_grammar,
                                   curr_graph,
                                   next_graph,
                                   next_time,
                                   mode='joint',
                                   mu=mu)
    indep_grammar = update_grammar(base_grammar,
                                   curr_graph,
                                   next_graph,
                                   next_time,
                                   mode='independent',
                                   mu=mu)
    return curr_time, next_time, p, base_grammar, joint_grammar, indep_grammar


def main(dataset, parallel, rewire, mu):
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = 'results/experiment_sequential_random/'
    mkdir(join(rootpath, resultspath))

    base_grammars: dict[tuple[int, float], VRG] = {}
    joint_grammars: dict[tuple[int, int, float], VRG] = {}
    indep_grammars: dict[tuple[int, int, float], VRG] = {}

    time_graph_pairs: list[tuple[int, nx.Graph]] = load_data(dataset)

    if parallel:
        results = Parallel(n_jobs=55)(
            delayed(experiment)(curr_time, curr_graph, next_time, next_graph, mu)
            for (curr_time, curr_graph), (next_time, next_graph)
            in tqdm(zip(time_graph_pairs[:-1], time_graph_pairs[1:]))
            for p in np.linspace(0, rewire, 10)
        )
    else:
        for p in np.linspace(0, rewire, 10):
            results = [experiment(curr_time, curr_graph, next_time, next_graph, p, mu)
                       for (curr_time, curr_graph), (next_time, next_graph)
                       in zip(time_graph_pairs[:-1], time_graph_pairs[1:])]

    for curr_time, next_time, p, base_grammar, joint_grammar, indep_grammar in results:
        base_grammars[(curr_time, p)] = base_grammar
        joint_grammars[(curr_time, next_time, p)] = joint_grammar
        indep_grammars[(curr_time, next_time, p)] = indep_grammar

    base_mdls = {time: grammar.calculate_cost()
                 for time, grammar in base_grammars.items()}
    joint_mdls = {(curr_time, next_time): grammar.calculate_cost()
                  for (curr_time, next_time), grammar in joint_grammars.items()}
    indep_mdls = {(curr_time, next_time): grammar.calculate_cost()
                  for (curr_time, next_time), grammar in indep_grammars.items()}

    joint_lls = {(curr_time, next_time): grammar.conditional_ll()
                 for (curr_time, next_time), grammar in joint_grammars.items()}
    indep_lls = {(curr_time, next_time): grammar.conditional_ll()
                 for (curr_time, next_time), grammar in indep_grammars.items()}

    with open(join(rootpath, resultspath, f'{dataset}_base.grammars'), 'wb') as outfile:
        pickle.dump(base_grammars, outfile)
    with open(join(rootpath, resultspath, f'{dataset}_joint.grammars'), 'wb') as outfile:
        pickle.dump(joint_grammars, outfile)
    with open(join(rootpath, resultspath, f'{dataset}_indep.grammars'), 'wb') as outfile:
        pickle.dump(indep_grammars, outfile)

    with open(join(rootpath, resultspath, f'{dataset}_base.mdls'), 'w') as outfile:
        for (time, p), mdl in base_mdls.items():
            outfile.write(f'{time},{p},{mdl}\n')
    with open(join(rootpath, resultspath, f'{dataset}_joint.mdls'), 'w') as outfile:
        for (curr_time, next_time, p), mdl in joint_mdls.items():
            outfile.write(f'{curr_time},{next_time},{p},{mdl}\n')
    with open(join(rootpath, resultspath, f'{dataset}_indep.mdls'), 'w') as outfile:
        for (curr_time, next_time, p), mdl in indep_mdls.items():
            outfile.write(f'{curr_time},{next_time},{p},{mdl}\n')

    with open(join(rootpath, resultspath, f'{dataset}_joint.lls'), 'w') as outfile:
        for (curr_time, next_time), ll in joint_lls.items():
            outfile.write(f'{curr_time},{next_time},{p},{ll}\n')
    with open(join(rootpath, resultspath, f'{dataset}_indep.lls'), 'w') as outfile:
        for (curr_time, next_time), ll in indep_lls.items():
            outfile.write(f'{curr_time},{next_time},{p},{ll}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset',
                        default='facebook-links',
                        dest='dataset',
                        type=str,
                        choices=['facebook-links', 'email-dnc', 'email-eucore', 'email-enron'],
                        help='select a dataset from ["facebook-links", "email-dnc", "email-eucore", "email-enron"]')
    parser.add_argument('-p', '--parallel',
                        action='store_true',
                        default=False,
                        dest='parallel',
                        help='run the experiment in parallel or not')
    parser.add_argument('-r', '--rewire',
                        default=4,
                        dest='rewire',
                        type=float,
                        help='the max percentage of edges to rewire')
    parser.add_argument('-m', '--mu',
                        default=4,
                        dest='mu',
                        type=int,
                        help='select a value for the μ hyperparameter for CNRG')
    args = parser.parse_args()
    main(args.dataset, args.parallel, args.rewire, args.mu)
