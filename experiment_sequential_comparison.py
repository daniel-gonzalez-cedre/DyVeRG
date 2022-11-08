import pickle
from os import getcwd
from os.path import join
from argparse import ArgumentParser

import sys
sys.path.append('./src/')

import git
from tqdm import tqdm
from networkx import Graph
from joblib import Parallel, delayed

from cnrg.VRG import VRG
from src.utils import mkdir, silence
from src.data import load_data
from src.bookkeeping import decompose
from src.update_grammar import update_grammar


def experiment(curr_time: int, curr_graph: Graph, next_time: int, next_graph: Graph, mu: int):
    with silence(enabled=False):
        base_grammar = decompose(curr_graph, time=curr_time, mu=mu)
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
    return curr_time, next_time, base_grammar, joint_grammar, indep_grammar


def main(dataset, parallel, mu):
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = 'results/experiment_sequential_comparison/'
    mkdir(join(rootpath, resultspath))

    base_grammars: dict[int, VRG] = {}
    joint_grammars: dict[tuple[int, int], VRG] = {}
    indep_grammars: dict[tuple[int, int], VRG] = {}

    time_graph_pairs: list[tuple[int, Graph]] = load_data(dataset)

    if parallel:
        results = Parallel(n_jobs=55)(
            delayed(experiment)(curr_time, curr_graph, next_time, next_graph, mu)
            for (curr_time, curr_graph), (next_time, next_graph)
            in tqdm(zip(time_graph_pairs[:-1], time_graph_pairs[1:]))
        )

        # for curr_time, next_time, base_grammar, joint_grammar, indep_grammar in results:
        #     base_grammars[curr_time] = base_grammar
        #     joint_grammars[(curr_time, next_time)] = joint_grammar
        #     indep_grammars[(curr_time, next_time)] = indep_grammar
    else:
        results = [experiment(curr_time, curr_graph, next_time, next_graph, mu)
                   for (curr_time, curr_graph), (next_time, next_graph)
                   in tqdm(zip(time_graph_pairs[:-1], time_graph_pairs[1:]))]

        # for (curr_time, curr_graph), (next_time, next_graph) in tqdm(zip(time_graph_pairs[:-1], time_graph_pairs[1:])):
        #     _, _, base_grammar, joint_grammar, indep_grammar = experiment(curr_time, curr_graph, next_time, next_graph, mu)

        #     base_grammars[curr_time] = base_grammar
        #     joint_grammars[(curr_time, next_time)] = joint_grammar
        #     indep_grammars[(curr_time, next_time)] = indep_grammar

    for curr_time, next_time, base_grammar, joint_grammar, indep_grammar in results:
        base_grammars[curr_time] = base_grammar
        joint_grammars[(curr_time, next_time)] = joint_grammar
        indep_grammars[(curr_time, next_time)] = indep_grammar

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
        for time, mdl in base_mdls.items():
            outfile.write(f'{time},{mdl}\n')
    with open(join(rootpath, resultspath, f'{dataset}_joint.mdls'), 'w') as outfile:
        for (curr_time, next_time), mdl in joint_mdls.items():
            outfile.write(f'{curr_time},{next_time},{mdl}\n')
    with open(join(rootpath, resultspath, f'{dataset}_indep.mdls'), 'w') as outfile:
        for (curr_time, next_time), mdl in indep_mdls.items():
            outfile.write(f'{curr_time},{next_time},{mdl}\n')

    with open(join(rootpath, resultspath, f'{dataset}_joint.lls'), 'w') as outfile:
        for (curr_time, next_time), ll in joint_lls.items():
            outfile.write(f'{curr_time},{next_time},{ll}\n')
    with open(join(rootpath, resultspath, f'{dataset}_indep.lls'), 'w') as outfile:
        for (curr_time, next_time), ll in indep_lls.items():
            outfile.write(f'{curr_time},{next_time},{ll}\n')


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
    parser.add_argument('-m', '--mu',
                        default=4,
                        dest='mu',
                        type=int,
                        help='select a value for the μ hyperparameter for CNRG')
    args = parser.parse_args()
    main(args.dataset, args.parallel, args.mu)
