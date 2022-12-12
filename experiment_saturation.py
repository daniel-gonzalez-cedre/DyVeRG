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


def experiment(trial: int, time_graph_pairs: list[tuple[int, nx.Graph]], mu: int) -> tuple[int, VRG, dict[tuple[int, int], VRG], dict[tuple[int, int], VRG]]:
    base_time, base_graph = time_graph_pairs[0]
    time, graph = time_graph_pairs[1]

    base_grammar = decompose(base_graph, time=base_time, mu=mu)

    prev_i_grammar = update_grammar(base_grammar, base_graph, graph, time, 'i')
    i_grammars = {(base_time, time): prev_i_grammar}

    prev_ja_grammar = update_grammar(base_grammar, base_graph, graph, time, 'ja')
    ja_grammars = {(base_time, time): prev_ja_grammar}

    prev_jb_grammar = update_grammar(base_grammar, base_graph, graph, time, 'jb')
    jb_grammars = {(base_time, time): prev_jb_grammar}

    cumulative_graph = nx.compose(base_graph, graph)
    for next_time, next_graph in time_graph_pairs[2:]:
        i_grammar = update_grammar(prev_i_grammar, cumulative_graph, next_graph, next_time, 'i')
        i_grammars[(time, next_time)] = i_grammar

        ja_grammar = update_grammar(prev_ja_grammar, cumulative_graph, next_graph, next_time, 'ja')
        ja_grammars[(time, next_time)] = ja_grammar

        jb_grammar = update_grammar(prev_jb_grammar, cumulative_graph, next_graph, next_time, 'jb')
        jb_grammars[(time, next_time)] = jb_grammar

        time = next_time
        prev_i_grammar = i_grammar
        prev_ja_grammar = ja_grammar
        prev_jb_grammar = jb_grammar
        cumulative_graph = nx.compose(cumulative_graph, next_graph)

    return trial, base_grammar, i_grammars, ja_grammars, jb_grammars


def main(pre_dispatch: bool):
    def tasks():
        for trial in range(1, args.n_trials + 1):
            yield (trial, time_graph_pairs, args.mu)

    def write(res):
        for trial, base_grammar, i_grammars, ja_grammars, jb_grammars in res:
            pickle.dump(base_grammar, base_gf)
            pickle.dump(i_grammars, i_gf)
            pickle.dump(ja_grammars, ja_gf)
            pickle.dump(jb_grammars, jb_gf)

            base_mdlf.write(f'{trial},{base_grammar.mdl}\n')
            for (curr_time, next_time), i_grammar in i_grammars.items():
                i_mdlf.write(f'{trial},{curr_time},{next_time},{i_grammar.mdl}\n')
                i_llf.write(f'{trial},{curr_time},{next_time},{i_grammar.ll}\n')
            for (curr_time, next_time), ja_grammar in ja_grammars.items():
                ja_mdlf.write(f'{trial},{curr_time},{next_time},{ja_grammar.mdl}\n')
                ja_llf.write(f'{trial},{curr_time},{next_time},{ja_grammar.ll}\n')
            for (curr_time, next_time), jb_grammar in jb_grammars.items():
                jb_mdlf.write(f'{trial},{curr_time},{next_time},{jb_grammar.mdl}\n')
                jb_llf.write(f'{trial},{curr_time},{next_time},{jb_grammar.ll}\n')

    time_graph_pairs: list[tuple[int, nx.Graph]] = load_data(args.dataset)

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
                delayed(experiment)(trial, time_graph_pairs, args.mu)
                for trial in range(1, args.n_trials + 1)
            )
        else:
            results = [experiment(trial, time_graph_pairs, args.mu)
                       for trial in range(1, args.n_trials + 1)]
        write(results)


# nice -n 10 python experiment_random.py email-enron -n 100 -p -j 25 -m 4
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset',
                        default='email-eucore',
                        type=str,
                        choices=['facebook-links', 'email-dnc', 'email-eucore', 'email-enron'],
                        help='select a dataset from ["facebook-links", "email-dnc", "email-eucore", "email-enron"]')
    parser.add_argument('-n', '--num',
                        default=10,
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
                        default=10,
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
    resultspath = 'results/experiment_saturation/'
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

    base_mdlf.write('trial,mdl\n')
    i_mdlf.write('trial,time1,time2,mdl\n')
    ja_mdlf.write('trial,time1,time2,mdl\n')
    jb_mdlf.write('trial,time1,time2,mdl\n')
    i_llf.write('trial,time1,time2,ll\n')
    ja_llf.write('trial,time1,time2,ll\n')
    jb_llf.write('trial,time1,time2,ll\n')

    main(pre_dispatch=args.pre_dispatch)

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
