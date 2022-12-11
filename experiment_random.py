import random
import pickle
from os import getcwd
from os.path import join
from argparse import ArgumentParser

import git
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

from cnrg.VRG import VRG
from src.utils import mkdir
from src.data import load_data
from src.decomposition import decompose
from src.adjoin_graph import update_grammar


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

    return h


def experiment(trial: int,
               curr_time: int, curr_graph: nx.Graph,
               next_time: int, next_graph: nx.Graph,
               p: float, mu: int) -> tuple[int, int, int, float, VRG, VRG, VRG, VRG]:
    base_grammar = decompose(curr_graph, time=curr_time, mu=mu)
    perturbed_graph = perturb_graph(next_graph, p)
    jagrammar = update_grammar(base_grammar,
                               curr_graph,
                               perturbed_graph,
                               next_time,
                               mode='ja')
    jbgrammar = update_grammar(base_grammar,
                               curr_graph,
                               perturbed_graph,
                               next_time,
                               mode='jb')
    igrammar = update_grammar(base_grammar,
                              curr_graph,
                              perturbed_graph,
                              next_time,
                              mode='i')
    return trial, curr_time, next_time, p, base_grammar, igrammar, jagrammar, jbgrammar


# TODO: implement saving intermediate results in case we need to stop the code
def main(dataset, rewire, delta, n_trials, parallel, n_jobs, mu):
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = 'results/experiment_random/'
    mkdir(join(rootpath, resultspath))

    base_grammars: dict[tuple[int, int, float], VRG] = {}
    igrammars: dict[tuple[int, int, int, float], VRG] = {}
    jagrammars: dict[tuple[int, int, int, float], VRG] = {}
    jbgrammars: dict[tuple[int, int, int, float], VRG] = {}

    time_graph_pairs: list[tuple[int, nx.Graph]] = load_data(dataset)

    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(experiment)(trial, curr_time, curr_graph, next_time, next_graph, p, mu)
            for (curr_time, curr_graph), (next_time, next_graph) in zip(time_graph_pairs[:-1], time_graph_pairs[1:])
            for p in np.linspace(0, rewire, delta)
            for trial in range(1, n_trials + 1)
        )
    else:
        results = [experiment(trial, curr_time, curr_graph, next_time, next_graph, p, mu)
                   for (curr_time, curr_graph), (next_time, next_graph) in zip(time_graph_pairs[:-1], time_graph_pairs[1:])
                   for p in np.linspace(0, rewire, delta)
                   for trial in range(1, n_trials + 1)]

    for trial, curr_time, next_time, p, base_grammar, igrammar, jagrammar, jbgrammar in results:
        base_grammars[(trial, curr_time, p)] = base_grammar
        igrammars[(trial, curr_time, next_time, p)] = igrammar
        jagrammars[(trial, curr_time, next_time, p)] = jagrammar
        jbgrammars[(trial, curr_time, next_time, p)] = jbgrammar

    with open(join(rootpath, resultspath, f'{dataset}_base.grammars'), 'wb') as base_file, \
         open(join(rootpath, resultspath, f'{dataset}_i.grammars'), 'wb') as ifile, \
         open(join(rootpath, resultspath, f'{dataset}_ja.grammars'), 'wb') as jafile, \
         open(join(rootpath, resultspath, f'{dataset}_jb.grammars'), 'wb') as jbfile:
        pickle.dump(base_grammars, base_file)
        pickle.dump(igrammars, ifile)
        pickle.dump(jagrammars, jafile)
        pickle.dump(jbgrammars, jbfile)

    with open(join(rootpath, resultspath, f'{dataset}_base.mdls'), 'w') as base_file, \
         open(join(rootpath, resultspath, f'{dataset}_i.mdls'), 'w') as ifile, \
         open(join(rootpath, resultspath, f'{dataset}_ja.mdls'), 'w') as jafile, \
         open(join(rootpath, resultspath, f'{dataset}_jb.mdls'), 'w') as jbfile:
        base_mdls = {key: grammar.mdl for key, grammar in base_grammars.items()}
        imdls = {key: grammar.mdl for key, grammar in igrammars.items()}
        jamdls = {key: grammar.mdl for key, grammar in jagrammars.items()}
        jbmdls = {key: grammar.mdl for key, grammar in jbgrammars.items()}

        base_file.write('trial,time,p,mdl\n')
        ifile.write('trial,time1,time2,p,mdl\n')
        jafile.write('trial,time1,time2,p,mdl\n')
        jbfile.write('trial,time1,time2,p,mdl\n')
        for (trial, time, p), mdl in base_mdls.items():
            base_file.write(f'{trial},{time},{p},{mdl}\n')
        for (trial, time1, time2, p), mdl in imdls.items():
            ifile.write(f'{trial},{time1},{time2},{p},{mdl}\n')
        for (trial, time1, time2, p), mdl in jamdls.items():
            jafile.write(f'{trial},{time1},{time2},{p},{mdl}\n')
        for (trial, time1, time2, p), mdl in jbmdls.items():
            jbfile.write(f'{trial},{time1},{time2},{p},{mdl}\n')

    with open(join(rootpath, resultspath, f'{dataset}_i.lls'), 'w') as ifile, \
         open(join(rootpath, resultspath, f'{dataset}_ja.lls'), 'w') as jafile, \
         open(join(rootpath, resultspath, f'{dataset}_jb.lls'), 'w') as jbfile:
        ills = {key: grammar.ll for key, grammar in igrammars.items()}
        jalls = {key: grammar.ll for key, grammar in jagrammars.items()}
        jblls = {key: grammar.ll for key, grammar in jbgrammars.items()}

        ifile.write('trial,time1,time2,p,ll\n')
        jafile.write('trial,time1,time2,p,ll\n')
        jbfile.write('trial,time1,time2,p,ll\n')
        for (trial, time1, time2, p), ll in ills.items():
            ifile.write(f'{trial},{time1},{time2},{p},{ll}\n')
        for (trial, time1, time2, p), ll in jalls.items():
            jafile.write(f'{trial},{time1},{time2},{p},{ll}\n')
        for (trial, time1, time2, p), ll in jblls.items():
            jbfile.write(f'{trial},{time1},{time2},{p},{ll}\n')


# python experiment_sequential_random.py [dataset] -d [delta] -r [rewire] -n [# trials] -p -j [# jobs] -m [mu]
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset',
                        default='email-eucore',
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
