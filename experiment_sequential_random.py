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
from src.utils import mkdir
from src.data import load_data
from src.decomposition import decompose
from src.graph_transitions import update_grammar


# g: the graph to perturb
# p: the percentage of the edges to rewire
def perturb_graph(g: nx.Graph, p: float = 0.01) -> nx.Graph:
    h = g.copy()

    if p == 0:
        return h

    num_edges = int(h.size() * p)
    edge_sample = random.sample(list(h.edges()), num_edges)  # casting to list to avoid deprecation warnings

    for old_u, old_v in edge_sample:
        h.remove_edge(old_u, old_v)
        u, v = random.sample(list(h.nodes()), 2)  # casting to list to avoid deprecation warnings

        while (u, v) in h.edges():
            u, v = random.sample(list(h.nodes()), 2)  # casting to list to avoid deprecation warnings

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
                                   perturbed_graph,
                                   next_time,
                                   mode='j')
    indep_grammar = update_grammar(base_grammar,
                                   curr_graph,
                                   perturbed_graph,
                                   next_time,
                                   mode='i')
    return curr_time, next_time, p, base_grammar, joint_grammar, indep_grammar


# TODO: implement saving intermediate results in case we need to stop the code
def main(dataset, rewire, delta, n_trials, parallel, n_jobs, mu):
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = 'results/experiment_sequential_random/'
    mkdir(join(rootpath, resultspath))

    base_grammars: dict[tuple[int, float], VRG] = {}
    joint_grammars: dict[tuple[int, int, float], VRG] = {}
    indep_grammars: dict[tuple[int, int, float], VRG] = {}

    time_graph_pairs: list[tuple[int, nx.Graph]] = load_data(dataset)

    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(experiment)(curr_time, curr_graph, next_time, next_graph, p, mu)
            for (curr_time, curr_graph), (next_time, next_graph)
            in zip(time_graph_pairs[:-1], time_graph_pairs[1:])
            for p in np.linspace(0, rewire, delta)
            for _ in range(n_trials)
        )
    else:
        results = [experiment(curr_time, curr_graph, next_time, next_graph, p, mu)
                   for (curr_time, curr_graph), (next_time, next_graph)
                   in zip(time_graph_pairs[:-1], time_graph_pairs[1:])
                   for p in np.linspace(0, rewire, delta)
                   for _ in range(n_trials)]

    for curr_time, next_time, p, base_grammar, joint_grammar, indep_grammar in results:
        if (curr_time, p) in base_grammars:
            base_grammars[(curr_time, p)] += [base_grammar]
        else:
            base_grammars[(curr_time, p)] = [base_grammar]

        if (curr_time, p) in joint_grammars:
            joint_grammars[(curr_time, next_time, p)] += [joint_grammar]
        else:
            joint_grammars[(curr_time, next_time, p)] = [joint_grammar]

        if (curr_time, p) in indep_grammars:
            indep_grammars[(curr_time, next_time, p)] += [indep_grammar]
        else:
            indep_grammars[(curr_time, next_time, p)] = [indep_grammar]

    base_mdls = {key: [grammar.mdl() for grammar in collection]
                 for key, collection in base_grammars.items()}

    joint_mdls = {key: [grammar.mdl() for grammar in collection]
                  for key, collection in joint_grammars.items()}
    indep_mdls = {key: [grammar.mdl() for grammar in collection]
                  for key, collection in indep_grammars.items()}

    joint_lls = {key: [grammar.ll() for grammar in collection]
                 for key, collection in joint_grammars.items()}
    indep_lls = {key: [grammar.ll() for grammar in collection]
                 for key, collection in indep_grammars.items()}

    with open(join(rootpath, resultspath, f'{dataset}_base.grammars'), 'wb') as outfile:
        pickle.dump(base_grammars, outfile)
    with open(join(rootpath, resultspath, f'{dataset}_joint.grammars'), 'wb') as outfile:
        pickle.dump(joint_grammars, outfile)
    with open(join(rootpath, resultspath, f'{dataset}_indep.grammars'), 'wb') as outfile:
        pickle.dump(indep_grammars, outfile)

    with open(join(rootpath, resultspath, f'{dataset}_base.mdls'), 'w') as outfile:
        outfile.write('time,p,mdl\n')
        for (time, p), mdls in base_mdls.items():
            for mdl in mdls:
                outfile.write(f'{time},{p},{mdl}\n')

    with open(join(rootpath, resultspath, f'{dataset}_joint.mdls'), 'w') as outfile:
        outfile.write('curr_time,next_time,p,mdl\n')
        for (curr_time, next_time, p), mdls in joint_mdls.items():
            for mdl in mdls:
                outfile.write(f'{curr_time},{next_time},{p},{mdl}\n')
    with open(join(rootpath, resultspath, f'{dataset}_indep.mdls'), 'w') as outfile:
        outfile.write('curr_time,next_time,p,mdl\n')
        for (curr_time, next_time, p), mdls in indep_mdls.items():
            for mdl in mdls:
                outfile.write(f'{curr_time},{next_time},{p},{mdl}\n')

    with open(join(rootpath, resultspath, f'{dataset}_joint.lls'), 'w') as outfile:
        outfile.write('curr_time,next_time,p,ll\n')
        for (curr_time, next_time, p), lls in joint_lls.items():
            for ll in lls:
                outfile.write(f'{curr_time},{next_time},{p},{ll}\n')
    with open(join(rootpath, resultspath, f'{dataset}_indep.lls'), 'w') as outfile:
        outfile.write('curr_time,next_time,p,ll\n')
        for (curr_time, next_time, p), lls in indep_lls.items():
            for ll in lls:
                outfile.write(f'{curr_time},{next_time},{p},{ll}\n')


# python experiment_sequential_random.py -d <<dataset>> -p -n <<# of jobs>> -r <<rewire %>> -m <<mu>>
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset',
                        default='email-eucore',
                        dest='dataset',
                        type=str,
                        choices=['facebook-links', 'email-dnc', 'email-eucore', 'email-enron'],
                        help='select a dataset from ["facebook-links", "email-dnc", "email-eucore", "email-enron"]')
    parser.add_argument('-r', '--rewire',
                        default=0.2,
                        dest='rewire',
                        type=float,
                        help='the max percentage of edges to rewire')
    parser.add_argument('-d', '--delta',
                        default=10,
                        dest='delta',
                        type=int,
                        help='the amount of intermediate rewires between 0 and `rewire`')
    parser.add_argument('-n', '--num',
                        default=5,
                        dest='n_trials',
                        type=int,
                        help='the number of times to run each experiment')
    parser.add_argument('-p', '--parallel',
                        action='store_true',
                        default=False,
                        dest='parallel',
                        help='run the experiment in parallel or not')
    parser.add_argument('-j', '--jobs',
                        default=40,
                        dest='n_jobs',
                        type=int,
                        help='the max number of parallel jobs to spawn')
    parser.add_argument('-m', '--mu',
                        default=4,
                        dest='mu',
                        type=int,
                        help='select a value for the μ hyperparameter for CNRG')
    args = parser.parse_args()
    main(args.dataset, args.rewire, args.delta, args.n_trials, args.parallel, args.n_jobs, args.mu)
