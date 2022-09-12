import random
import sys

sys.path.append('/g/g15/cedre/cnrg')
sys.path.append('/g/g15/cedre/gonzalez_mrhyde')

import numpy as np

from data_scripts import read_data
from rule_to_rule_scripts import decompose

if __name__ == '__main__':
    dataname = 'fb-messages'
    graphs, _ = read_data(dataname=dataname)
    _, grammar = decompose(graphs[0])
    # rule.compute_canon_matrix()
    # print(rule.canon_graph)
    # print(rule.canon_matrix)

    for rule in grammar.rule_list:
        try:
            rule.compute_canon_matrix()
        except ValueError as e:
            print(e)
            print(rule.graph.nodes(data=True))
            print(rule.graph.edges(data=True))
            exit()

    rule, = random.sample(grammar.rule_list, 1)

    for other_rule in grammar.rule_list:
        if rule.fast_eq(other_rule):
        # if rule.lhs == other_rule.lhs and np.array_equal(rule.canon_matrix, other_rule.canon_matrix):
            print(rule == other_rule)

    pass
