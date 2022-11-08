import sys
import pickle
# from os.path import join

sys.path.append('..')

# import numpy as np
from networkx import Graph
from joblib import Parallel, delayed

from cnrg.VRG import VRG as VRG
# from cnrg.LightMultiGraph import LightMultiGraph as LightMultiGraph

sys.path.append('../src')
# from data import load_data
# from data_old import read_data
from data import load_data
from bookkeeping import decompose
from update_grammar import update_grammar


def experiment(lookback, idx, home_graph, away_graph, mu):
    base_grammar = decompose(home_graph, mu=mu)
    joint_grammar = update_grammar(base_grammar,
                                   home_graph,
                                   away_graph,
                                   mode='joint')
    indep_grammar = update_grammar(base_grammar,
                                   home_graph,
                                   away_graph,
                                   mode='independent')

    return lookback, idx, base_grammar, joint_grammar, indep_grammar


def main():
    parallel = True

    mu = 4
    dataname = 'fb-messages'
    # graphs, years = read_data(dataname=dataname, lookback=lookback)

    dataname = 'fb-messages'
    lookbacks = list(range(11))

    base_grammars: dict[int, dict[int, VRG]] = {lookback: {} for lookback in lookbacks}
    joint_grammars: dict[int, dict[int, VRG]] = {lookback: {} for lookback in lookbacks}
    indep_grammars: dict[int, dict[int, VRG]] = {lookback: {} for lookback in lookbacks}

    base_mdls: dict[int, dict[int, float]] = {lookback: {} for lookback in lookbacks}
    joint_mdls: dict[int, dict[int, float]] = {lookback: {} for lookback in lookbacks}
    joint_lls: dict[int, dict[int, float]] = {lookback: {} for lookback in lookbacks}
    indep_mdls: dict[int, dict[int, float]] = {lookback: {} for lookback in lookbacks}
    indep_lls: dict[int, dict[int, float]] = {lookback: {} for lookback in lookbacks}

    if parallel:
        graph_tensor: dict[int, Graph] = {lookback: graphs
                                          for lookback in lookbacks
                                          for graphs, _ in [read_data(dataname=dataname, lookback=lookback)]}

        results = Parallel(n_jobs=55)(
            delayed(experiment)(lookback, idx, home_graph, away_graph, mu)
            for lookback in lookbacks
            for idx, (home_graph, away_graph) in enumerate(zip(graph_tensor[lookback][:-1], graph_tensor[lookback][1:]))
        )

        for lookback, idx, base_grammar, joint_grammar, indep_grammar in results:
            base_grammars[lookback][idx] = base_grammar
            joint_grammars[lookback][idx] = joint_grammar
            indep_grammars[lookback][idx] = indep_grammar

            base_mdls[lookback][idx] = base_grammar.calculate_cost()

            joint_mdls[lookback][idx] = joint_grammar.calculate_cost()
            joint_lls[lookback][idx] = joint_grammar.conditional_ll()

            indep_mdls[lookback][idx] = indep_grammar.calculate_cost()
            indep_lls[lookback][idx] = indep_grammar.conditional_ll()

        for lookback in lookbacks:
            base_grammar = decompose(graph_tensor[lookback][-1], mu=mu)
            base_grammars[lookback][len(base_grammars[lookback])] = base_grammar
            base_mdls[lookback][len(base_mdls[lookback])] = base_grammar.calculate_cost()
    else:
        for lookback in lookbacks:
            graphs, years = read_data(dataname=dataname, lookback=lookback)

            for idx, (home_graph, away_graph) in enumerate(zip(graphs[:-1], graphs[1:])):
                base_grammars[lookback][idx] = decompose(home_graph, mu=mu)
                joint_grammars[lookback][idx] = update_grammar(base_grammars[lookback],
                                                               home_graph,
                                                               away_graph,
                                                               mode='joint')
                indep_grammars[lookback][idx] = update_grammar(base_grammars[lookback],
                                                               home_graph,
                                                               away_graph,
                                                               mode='independent')

                base_mdls[lookback][idx] = base_grammars[lookback][idx].calculate_cost()

                joint_mdls[lookback][idx] = joint_grammars[lookback][idx].calculate_cost()
                joint_lls[lookback][idx] = joint_grammars[lookback][idx].conditional_ll()

                indep_mdls[lookback][idx] = indep_grammars[lookback][idx].calculate_cost()
                indep_lls[lookback][idx] = indep_grammars[lookback][idx].conditional_ll()

            base_grammar = decompose(graphs[-1], mu=mu)
            base_grammars[lookback][idx + 1] = base_grammar
            base_mdls[lookback][idx + 1] = base_grammar.calculate_cost()

    with open(f'../results/experiment_sequential/{dataname}_base.grammars', 'wb') as outfile:
        pickle.dump(base_grammars, outfile)
    with open(f'../results/experiment_sequential/{dataname}_joint.grammars', 'wb') as outfile:
        pickle.dump(joint_grammars, outfile)
    with open(f'../results/experiment_sequential/{dataname}_indep.grammars', 'wb') as outfile:
        pickle.dump(indep_grammars, outfile)

    with open(f'../results/experiment_sequential/{dataname}_base.mdls', 'wb') as outfile:
        pickle.dump(base_mdls, outfile)
    with open(f'../results/experiment_sequential/{dataname}_joint.mdls', 'wb') as outfile:
        pickle.dump(joint_mdls, outfile)
    with open(f'../results/experiment_sequential/{dataname}_joint.lls', 'wb') as outfile:
        pickle.dump(joint_lls, outfile)
    with open(f'../results/experiment_sequential/{dataname}_indep.mdls', 'wb') as outfile:
        pickle.dump(indep_mdls, outfile)
    with open(f'../results/experiment_sequential/{dataname}_indep.lls', 'wb') as outfile:
        pickle.dump(indep_lls, outfile)
    return


if __name__ == '__main__':
    main()
