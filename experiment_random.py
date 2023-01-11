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
    i_grammar = update_grammar(base_grammar, curr_graph, perturbed_graph, curr_time, next_time, 'i')
    ja_grammar = update_grammar(base_grammar, curr_graph, perturbed_graph, curr_time, next_time, 'ja')
    jb_grammar = update_grammar(base_grammar, curr_graph, perturbed_graph, curr_time, next_time, 'jb')

    # save the grammars
    with open(pathprefix + f'_base_{curr_time}_{trial}.grammars', 'wb') as basefile:
        pickle.dump(base_grammar, basefile)
    with open(pathprefix + f'_i_{curr_time}_{next_time}_{trial}.grammars', 'wb') as ifile:
        pickle.dump(i_grammar, ifile)
    with open(pathprefix + f'_ja_{curr_time}_{next_time}_{trial}.grammars', 'wb') as jafile:
        pickle.dump(ja_grammar, jafile)
    with open(pathprefix + f'_jb_{curr_time}_{next_time}_{trial}.grammars', 'wb') as jbfile:
        pickle.dump(jb_grammar, jbfile)

    return (trial, curr_time, next_time, p,
            base_grammar.mdl, i_grammar.mdl, ja_grammar.mdl, jb_grammar.mdl,
            i_grammar.ll(next_time, prior=curr_time), ja_grammar.ll(next_time, prior=curr_time), jb_grammar.ll(next_time, prior=curr_time))
    # return trial, curr_time, next_time, p, base_grammar, i_grammar, ja_grammar, jb_grammar


def main(pre_dispatch: bool):
    def tasks():
        for trial in range(1, args.n_trials + 1):
            for p in np.linspace(0, args.rewire, args.delta):
                for curr_time, next_time in zip(times[:-1], times[1:]):
                    yield (trial, curr_time, next_time, graphs[curr_time], graphs[next_time], p, args.mu)

    time_graph_pairs: list[tuple[int, nx.Graph]] = load_data(args.dataset)
    times: list[int] = [t for t, _ in time_graph_pairs]
    graphs: dict[int, nx.Graph] = {t: g for t, g in time_graph_pairs}  # pylint: disable=unnecessary-comprehension

    # run the experiments
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

    # write the output
    with open(pathprefix + '_base.mdls', 'w') as basemdlfile, \
         open(pathprefix + '_i.mdls', 'w') as imdlfile, \
         open(pathprefix + '_ja.mdls', 'w') as jamdlfile, \
         open(pathprefix + '_jb.mdls', 'w') as jbmdlfile, \
         open(pathprefix + '_i.lls', 'w') as illfile, \
         open(pathprefix + '_ja.lls', 'w') as jallfile, \
         open(pathprefix + '_jb.lls', 'w') as jbllfile:
        basemdlfile.write('trial,time,p,mdl\n')
        imdlfile.write('trial,time1,time2,p,mdl\n')
        jamdlfile.write('trial,time1,time2,p,mdl\n')
        jbmdlfile.write('trial,time1,time2,p,mdl\n')
        illfile.write('trial,time1,time2,p,ll\n')
        jallfile.write('trial,time1,time2,p,ll\n')
        jbllfile.write('trial,time1,time2,p,ll\n')

        for trial, curr_time, next_time, p, base_mdl, i_mdl, ja_mdl, jb_mdl, i_ll, ja_ll, jb_ll in results:
            basemdlfile.write(f'{trial},{curr_time},{p},{base_mdl}\n')
            imdlfile.write(f'{trial},{curr_time},{next_time},{p},{i_mdl}\n')
            jamdlfile.write(f'{trial},{curr_time},{next_time},{p},{ja_mdl}\n')
            jbmdlfile.write(f'{trial},{curr_time},{next_time},{p},{jb_mdl}\n')

            illfile.write(f'{trial},{curr_time},{next_time},{p},{i_ll}\n')
            jallfile.write(f'{trial},{curr_time},{next_time},{p},{ja_ll}\n')
            jbllfile.write(f'{trial},{curr_time},{next_time},{p},{jb_ll}\n')


# nice -n 10 python experiment_random.py email-eucore -r 0.25 -d 25 -n 25 -p -j 40 -m 4
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
    parser.add_argument('-l', '--predispatch',
                        action='store_true',
                        default=False,
                        dest='pre_dispatch',
                        help='pre-dispatch the jobs or not')
    parser.add_argument('-m', '--mu',
                        default=4,
                        dest='mu',
                        type=int,
                        help='select a value for the μ hyperparameter for CNRG')
    args = parser.parse_args()

    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = 'results/experiment_random/'
    pathprefix = join(rootpath, resultspath, f'{args.dataset}_{args.mu}')
    mkdir(join(rootpath, resultspath))

    # pylint: disable=consider-using-with
    main(pre_dispatch=args.pre_dispatch)
