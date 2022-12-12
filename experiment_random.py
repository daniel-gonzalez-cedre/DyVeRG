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


def experiment(trial: int, curr_time: int, next_time: int,
               curr_graph: nx.Graph, next_graph: nx.Graph,
               p: float, mu: int) -> tuple[int, int, int, float, VRG, VRG, VRG, VRG]:
    perturbed_graph = perturb_graph(next_graph, p)

    base_grammar = decompose(curr_graph, time=curr_time, mu=mu)
    i_grammar = update_grammar(base_grammar, curr_graph, perturbed_graph, next_time, 'i')
    ja_grammar = update_grammar(base_grammar, curr_graph, perturbed_graph, next_time, 'ja')
    jb_grammar = update_grammar(base_grammar, curr_graph, perturbed_graph, next_time, 'jb')
    return trial, curr_time, next_time, p, base_grammar, i_grammar, ja_grammar, jb_grammar


def main(pre_dispatch: bool):
    def tasks():
        for trial in range(1, args.n_trials + 1):
            for p in np.linspace(0, args.rewire, args.delta):
                for curr_time, next_time in zip(times[:-1], times[1:]):
                    yield (trial, curr_time, next_time, graphs[curr_time], graphs[next_time], p, args.mu)

    def write(res):
        for trial, curr_time, next_time, p, base_grammar, i_grammar, ja_grammar, jb_grammar in res:
            pickle.dump(base_grammar, base_gf)
            pickle.dump(i_grammar, i_gf)
            pickle.dump(ja_grammar, ja_gf)
            pickle.dump(jb_grammar, jb_gf)

            base_mdlf.write(f'{trial},{curr_time},{p},{base_grammar.mdl}\n')
            i_mdlf.write(f'{trial},{curr_time},{next_time},{p},{i_grammar.mdl}\n')
            ja_mdlf.write(f'{trial},{curr_time},{next_time},{p},{ja_grammar.mdl}\n')
            jb_mdlf.write(f'{trial},{curr_time},{next_time},{p},{jb_grammar.mdl}\n')

            i_llf.write(f'{trial},{curr_time},{next_time},{p},{i_grammar.ll}\n')
            ja_llf.write(f'{trial},{curr_time},{next_time},{p},{ja_grammar.ll}\n')
            jb_llf.write(f'{trial},{curr_time},{next_time},{p},{jb_grammar.ll}\n')

    time_graph_pairs: list[tuple[int, nx.Graph]] = load_data(args.dataset)
    times: list[int] = [t for t, _ in time_graph_pairs]
    graphs: dict[int, nx.Graph] = {t: g for t, g in time_graph_pairs}  # pylint: disable=unnecessary-comprehension

    if args.batch:
        batch = []
        batch_size = 4 * args.n_jobs
        for num, task in enumerate(tasks()):
            batch.append(task)

            if num % batch_size == 0:
                if args.parallel:
                    results = Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch='all' if pre_dispatch else '2 * n_jobs')(
                        delayed(experiment)(*task) for task in batch
                    )
                else:
                    results = [experiment(*task) for task in batch]

                batch = []
                write(results)

        # clean up last batch
        if args.parallel:
            results = Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch='all' if pre_dispatch else '2 * n_jobs')(
                delayed(experiment)(*task) for task in batch
            )
        else:
            results = [experiment(*task) for task in batch]
        write(results)
    else:
        if args.parallel:
            results = Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch='all' if pre_dispatch else '2 * n_jobs')(
                delayed(experiment)(trial, curr_time, next_time, graphs[curr_time], graphs[next_time], p, args.mu)
                for curr_time, next_time in zip(times[:-1], times[1:])
                for p in np.linspace(0, args.rewire, args.delta)
                for trial in range(1, args.n_trials + 1)
            )
        else:
            results = [experiment(trial, curr_time, next_time, graphs[curr_time], graphs[next_time], p, args.mu)
                       for (curr_time, curr_graph), (next_time, next_graph) in zip(time_graph_pairs[:-1], time_graph_pairs[1:])
                       for p in np.linspace(0, args.rewire, args.delta)
                       for trial in range(1, args.n_trials + 1)]
        write(results)


# nice -n 10 python experiment_random.py email-enron -r 0.25 -d 25 -n 25 -p -j 40 -m 4
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset',
                        default='email-eucore',
                        type=str,
                        choices=['facebook-links', 'email-dnc', 'email-eucore', 'email-enron'],
                        help='select a dataset from ["facebook-links", "email-dnc", "email-eucore", "email-enron"]')
    parser.add_argument('-r', '--rewire',
                        default=0.25,
                        dest='rewire',
                        type=float,
                        help='the max percentage of edges to rewire')
    parser.add_argument('-d', '--delta',
                        default=25,
                        dest='delta',
                        type=int,
                        help='the amount of intermediate rewires between 0 and `rewire`')
    parser.add_argument('-n', '--num',
                        default=5,
                        dest='n_trials',
                        type=int,
                        help='the number of times to run each experiment')
    parser.add_argument('-b', '--batch',
                        action='store_true',
                        default=False,
                        dest='batch',
                        help='batch the compute tasks or not')
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

    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = 'results/experiment_random/'
    mkdir(join(rootpath, resultspath))

    # pylint: disable=consider-using-with
    base_gf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_base.grammars'), 'wb')
    i_gf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_i.grammars'), 'wb')
    ja_gf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_ja.grammars'), 'wb')
    jb_gf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_jb.grammars'), 'wb')
    base_mdlf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_base.mdls'), 'w')
    i_mdlf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_i.mdls'), 'w')
    ja_mdlf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_ja.mdls'), 'w')
    jb_mdlf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_jb.mdls'), 'w')
    i_llf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_i.lls'), 'w')
    ja_llf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_ja.lls'), 'w')
    jb_llf = open(join(rootpath, resultspath, f'{args.dataset}_{args.mu}_jb.lls'), 'w')

    base_mdlf.write('trial,time,p,mdl\n')
    i_mdlf.write('trial,time1,time2,p,mdl\n')
    ja_mdlf.write('trial,time1,time2,p,mdl\n')
    jb_mdlf.write('trial,time1,time2,p,mdl\n')
    i_llf.write('trial,time1,time2,p,ll\n')
    ja_llf.write('trial,time1,time2,p,ll\n')
    jb_llf.write('trial,time1,time2,p,ll\n')

    main(pre_dispatch=True)

    base_gf.close()
    i_gf.close()
    ja_gf.close()
    jb_gf.close()
    base_mdlf.close()
    i_mdlf.close()
    ja_mdlf.close()
    jb_mdlf.close()
    i_llf.close()
    ja_llf.close()
    jb_llf.close()
