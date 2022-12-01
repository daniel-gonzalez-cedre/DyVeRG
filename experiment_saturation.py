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
from src.adjoin_graph import update_grammar


def experiment(trial: int, time_graph_pairs: list[tuple[int, nx.Graph]], mu: int) -> tuple[int, VRG, dict[tuple[int, int], VRG], dict[tuple[int, int], VRG]]:
    base_time, base_graph = time_graph_pairs[0]
    time, graph = time_graph_pairs[1]

    base_grammar = decompose(base_graph, time=base_time, mu=mu)

    prev_joint_grammar = update_grammar(base_grammar, base_graph, graph, time, mode='j')
    joint_grammars = {(base_time, time): prev_joint_grammar}

    prev_indep_grammar = update_grammar(base_grammar, base_graph, graph, time, mode='i')
    indep_grammars = {(base_time, time): prev_indep_grammar}

    cumulative_graph = nx.compose(base_graph, graph)
    for next_time, next_graph in time_graph_pairs[2:]:
        joint_grammar = update_grammar(prev_joint_grammar, cumulative_graph, next_graph, next_time, mode='j')
        joint_grammars[(time, next_time)] = joint_grammar

        indep_grammar = update_grammar(prev_indep_grammar, cumulative_graph, next_graph, next_time, mode='i')
        indep_grammars[(time, next_time)] = indep_grammar

        time = next_time
        prev_joint_grammar = joint_grammar
        prev_indep_grammar = indep_grammar
        cumulative_graph = nx.compose(cumulative_graph, next_graph)

    return trial, base_grammar, joint_grammars, indep_grammars


# TODO: implement saving intermediate results in case we need to stop the code
def main(dataset, n_trials, parallel, n_jobs, mu):
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = 'results/experiment_saturation/'
    mkdir(join(rootpath, resultspath))

    base_grammars: dict[int, VRG] = {}
    joint_grammars: dict[int, dict[tuple[int, int], VRG]] = {}
    indep_grammars: dict[int, dict[tuple[int, int], VRG]] = {}

    time_graph_pairs: list[tuple[int, nx.Graph]] = load_data(dataset)

    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(experiment)(trial, time_graph_pairs[:3], mu)
            for trial in range(n_trials)
        )
    else:
        results = [experiment(trial, time_graph_pairs[:3], mu)
                   for trial in range(n_trials)]

    for trial, base_grammar, joint_dict, indep_dict in results:
        base_grammars[trial] = base_grammar
        joint_grammars[trial] = joint_dict
        indep_grammars[trial] = indep_dict

    base_mdls = {key: grammar.mdl
                 for key, grammar in base_grammars.items()}

    joint_mdls = {trial: {(t1, t2): grammar.mdl for (t1, t2), grammar in joint_dict.items()}
                  for trial, joint_dict in joint_grammars.items()}
    indep_mdls = {trial: {(t1, t2): grammar.mdl for (t1, t2), grammar in indep_dict.items()}
                  for trial, indep_dict in indep_grammars.items()}

    joint_lls = {trial: {(t1, t2): grammar.ll for (t1, t2), grammar in joint_dict.items()}
                 for trial, joint_dict in joint_grammars.items()}
    indep_lls = {trial: {(t1, t2): grammar.ll for (t1, t2), grammar in indep_dict.items()}
                 for trial, indep_dict in indep_grammars.items()}

    with open(join(rootpath, resultspath, f'{dataset}_base.grammars'), 'wb') as outfile:
        pickle.dump(base_grammars, outfile)
    with open(join(rootpath, resultspath, f'{dataset}_joint.grammars'), 'wb') as outfile:
        pickle.dump(joint_grammars, outfile)
    with open(join(rootpath, resultspath, f'{dataset}_indep.grammars'), 'wb') as outfile:
        pickle.dump(indep_grammars, outfile)

    with open(join(rootpath, resultspath, f'{dataset}_base.mdls'), 'w') as outfile:
        outfile.write('trial,mdl\n')
        for trial, mdl in base_mdls.items():
            outfile.write(f'{trial},{mdl}\n')

    with open(join(rootpath, resultspath, f'{dataset}_joint.mdls'), 'w') as outfile:
        outfile.write('trial,t1,t2,mdl\n')
        for trial, mdls in joint_mdls.items():
            for (t1, t2), mdl in mdls.items():
                outfile.write(f'{trial},{t1},{t2},{mdl}\n')
    with open(join(rootpath, resultspath, f'{dataset}_indep.mdls'), 'w') as outfile:
        outfile.write('trial,t1,t2,mdl\n')
        for trial, mdls in indep_mdls.items():
            for (t1, t2), mdl in mdls.items():
                outfile.write(f'{trial},{t1},{t2},{mdl}\n')

    with open(join(rootpath, resultspath, f'{dataset}_joint.lls'), 'w') as outfile:
        outfile.write('trial,t1,t2,ll\n')
        for trial, lls in joint_lls.items():
            for (t1, t2), ll in lls.items():
                outfile.write(f'{trial},{t1},{t2},{ll}\n')
    with open(join(rootpath, resultspath, f'{dataset}_indep.lls'), 'w') as outfile:
        outfile.write('trial,t1,t2,ll\n')
        for trial, lls in indep_lls.items():
            for (t1, t2), ll in lls.items():
                outfile.write(f'{trial},{t1},{t2},{ll}\n')


# python experiment_sequential_random.py [dataset] -d [delta] -r [rewire] -n [# trials] -p -j [# jobs] -m [mu]
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset',
                        default='email-eucore',
                        type=str,
                        choices=['facebook-links', 'email-dnc', 'email-eucore', 'email-enron'],
                        help='select a dataset from ["facebook-links", "email-dnc", "email-eucore", "email-enron"]')
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
    main(args.dataset, args.n_trials, args.parallel, args.n_jobs, args.mu)
