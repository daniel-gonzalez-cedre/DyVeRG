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
    igrammar = update_grammar(base_grammar,
                              curr_graph,
                              perturbed_graph,
                              'i',
                              next_time)
    jagrammar = update_grammar(base_grammar,
                               curr_graph,
                               perturbed_graph,
                               'ja',
                               next_time)
    jbgrammar = update_grammar(base_grammar,
                               curr_graph,
                               perturbed_graph,
                               'jb',
                               next_time)
    return trial, curr_time, next_time, p, base_grammar, igrammar, jagrammar, jbgrammar


def write(base_gf, i_gf, ja_gf, jb_gf,
          base_mdlf, i_mdlf, ja_mdlf, jb_mdlf,
          i_llf, ja_llf, jb_llf,
          results):
    for trial, curr_time, next_time, p, base_grammar, igrammar, jagrammar, jbgrammar in results:
        pickle.dump(base_grammar, base_gf)
        pickle.dump(igrammar, i_gf)
        pickle.dump(jagrammar, ja_gf)
        pickle.dump(jbgrammar, jb_gf)

        base_mdlf.write(f'{trial},{curr_time},{p},{base_grammar.mdl}\n')
        i_mdlf.write(f'{trial},{curr_time},{next_time},{p},{igrammar.mdl}\n')
        ja_mdlf.write(f'{trial},{curr_time},{next_time},{p},{jagrammar.mdl}\n')
        jb_mdlf.write(f'{trial},{curr_time},{next_time},{p},{jbgrammar.mdl}\n')

        i_llf.write(f'{trial},{curr_time},{next_time},{p},{igrammar.ll}\n')
        ja_llf.write(f'{trial},{curr_time},{next_time},{p},{jagrammar.ll}\n')
        jb_llf.write(f'{trial},{curr_time},{next_time},{p},{jbgrammar.ll}\n')


def main(dataset, rewire, delta, n_trials, parallel, n_jobs, mu):
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = 'results/experiment_random/'
    mkdir(join(rootpath, resultspath))

    time_graph_pairs: list[tuple[int, nx.Graph]] = load_data(dataset)

    with open(join(rootpath, resultspath, f'{dataset}_base.grammars'), 'wb') as base_gf, \
         open(join(rootpath, resultspath, f'{dataset}_i.grammars'), 'wb') as i_gf, \
         open(join(rootpath, resultspath, f'{dataset}_ja.grammars'), 'wb') as ja_gf, \
         open(join(rootpath, resultspath, f'{dataset}_jb.grammars'), 'wb') as jb_gf, \
         open(join(rootpath, resultspath, f'{dataset}_base.mdls'), 'w') as base_mdlf, \
         open(join(rootpath, resultspath, f'{dataset}_i.mdls'), 'w') as i_mdlf, \
         open(join(rootpath, resultspath, f'{dataset}_ja.mdls'), 'w') as ja_mdlf, \
         open(join(rootpath, resultspath, f'{dataset}_jb.mdls'), 'w') as jb_mdlf, \
         open(join(rootpath, resultspath, f'{dataset}_i.lls'), 'w') as i_llf, \
         open(join(rootpath, resultspath, f'{dataset}_ja.lls'), 'w') as ja_llf, \
         open(join(rootpath, resultspath, f'{dataset}_jb.lls'), 'w') as jb_llf:
        base_mdlf.write('trial,time,p,mdl\n')
        i_mdlf.write('trial,time1,time2,p,mdl\n')
        ja_mdlf.write('trial,time1,time2,p,mdl\n')
        jb_mdlf.write('trial,time1,time2,p,mdl\n')
        i_llf.write('trial,time1,time2,p,ll\n')
        ja_llf.write('trial,time1,time2,p,ll\n')
        jb_llf.write('trial,time1,time2,p,ll\n')

        if parallel:
            batch_size = 2 * n_jobs
            tasks = [(trial, curr_time, curr_graph, next_time, next_graph, p, mu)
                     for (curr_time, curr_graph), (next_time, next_graph) in zip(time_graph_pairs[:-1], time_graph_pairs[1:])
                     for p in np.linspace(0, rewire, delta)
                     for trial in range(1, n_trials + 1)]
            n_batches = len(tasks) // batch_size
            batches = [batch for start in range(n_batches + 1)
                       if (batch := tasks[(batch_size * start):(batch_size * start + batch_size)]) != []]

            for batch in batches:
                results = Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(experiment)(task) for task in batch
                )
                write(base_gf, i_gf, ja_gf, jb_gf,
                      base_mdlf, i_mdlf, ja_mdlf, jb_mdlf,
                      i_llf, ja_llf, jb_llf,
                      results)
        else:
            batch_size = 2 * n_jobs
            tasks = [(trial, curr_time, curr_graph, next_time, next_graph, p, mu)
                     for (curr_time, curr_graph), (next_time, next_graph) in zip(time_graph_pairs[:-1], time_graph_pairs[1:])
                     for p in np.linspace(0, rewire, delta)
                     for trial in range(1, n_trials + 1)]
            n_batches = len(tasks) // batch_size
            batches = [batch for start in range(n_batches + 1)
                       if (batch := tasks[(batch_size * start):(batch_size * start + batch_size)]) != []]

            for batch in batches:
                results = [experiment(trial, curr_time, curr_graph, next_time, next_graph, p, mu)
                           for (curr_time, curr_graph), (next_time, next_graph) in zip(time_graph_pairs[:-1], time_graph_pairs[1:])
                           for p in np.linspace(0, rewire, delta)
                           for trial in range(1, n_trials + 1)]
                write(base_gf, i_gf, ja_gf, jb_gf,
                      base_mdlf, i_mdlf, ja_mdlf, jb_mdlf,
                      i_llf, ja_llf, jb_llf,
                      results)


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
