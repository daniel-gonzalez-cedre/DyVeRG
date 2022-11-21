import random
import sys
from typing import List, Set, Union, Dict, Any, Tuple

import pynauty as pn
import networkx as nx

sys.path.append('/g/g15/cedre/cnrg')

from cnrg.LightMultiGraph import LightMultiGraph as LMG

from data_scripts import read_data, create_graphs
from rule_to_rule_scripts import convert_LMG, decompose, ancestor, common_ancestor
from rule_to_rule_scripts import update_grammar_independent, update_rule_case1, update_rule_case2


def canon_matrix(g: LMG):
    pass


def canon_node_order(g: LMG) -> Tuple[List[int], Dict[Any, int]]:
    g_simple = lmg_to_simple(g)
    g_relabeled = nx.convert_node_labels_to_integers(g_simple)
    g_pynauty = simple_to_nauty(g_relabeled)

    canon_order = pn.canon_label(g_pynauty)  # need to also recover the old node labels

    canon_mapping = {}
    for v, d in g_relabeled.nodes(data=True):
        canon_mapping[d['old_name']] = canon_order[v]

    return canon_order, canon_mapping


def lmg_to_simple(g: LMG) -> nx.Graph:
    h = nx.Graph(g)
    k = h.order()

    for u, v, d in g.edges(data=True):
        h.add_edge(u, v)

        for _ in range(1, d['weight']):
            h.add_nodes_from([k, k + 1])

            h.add_edges_from([(n, k) for n in h.neighbors(u) if n != v])
            h.add_edges_from([(n, k + 1) for n in h.neighbors(v) if n != u])

            h.add_edge(u, k, duplicate=True)
            h.add_edge(v, k + 1, duplicate=True)

            k = h.order()

    return h


def multi_to_simple(g: nx.MultiGraph) -> nx.Graph:
    h = nx.Graph(g)
    k = h.order()

    for u, v in g.edges():
        if (u, v) in h.edges():
            # (k) is a duplicate of u
            # (k + 1) is a duplicate of v
            h.add_nodes_from([k, k + 1])

            for key, val in g.nodes[u].items():
                h.nodes[k][key] = val

            for key, val in g.nodes[v].items():
                h.nodes[k + 1][key] = val

            h.add_edges_from([(n, k) for n in h.neighbors(u) if n != v])
            h.add_edges_from([(n, k + 1) for n in h.neighbors(v) if n != u])

            h.add_edge(u, k, duplicate=True)
            h.add_edge(v, k + 1, duplicate=True)

            k = h.order()
        else:
            h.add_edge(u, v)

    return h


# the input graph g must have its nodes named [0, 1, ... g.order() - 1]
def simple_to_nauty(g: Union[LMG, nx.Graph]) -> pn.Graph:
    # g = nx.convert_node_labels_to_integers(g)
    h = pn.Graph(g.order())
    colors: List[Set] = []

    for u, v in g.edges():
        h.connect_vertex(u, [v])

    # color the duplicate nodes with the same color
    # duplicate nodes were added when converting multigraph -> graph
    for u, v, d in g.edges(data=True):
        if 'duplicate' in d and d['duplicate']:
            novel_color = True

            for color in colors:
                if u in color or v in color:
                    color |= {u, v}
                    novel_color = False

            if novel_color:
                colors.append({u, v})

    # color the NTSs according to size
    for v, d in g.nodes(data=True):
        if 'label' in d:
            novel_color = True

            for color in colors:
                try:
                    nts = next(iter(color))  # pick an arbitrary representative
                    size = g.nodes[nts]['label']

                    if d['label'] == size:
                        color.add(v)
                        novel_color = False
                        break
                except KeyError:
                    pass

            if novel_color:
                colors.append({v})

    h.set_vertex_coloring(colors)

    return h


# some basic testing
if __name__ == '__main__':
    dataname = 'fb-messages'
    graphs, _ = read_data(dataname=dataname)
    _, grammar = decompose(graphs[0])
    rule, = random.sample(grammar.rule_list, 1)
    # g = rule.graph
    # g_lmg = lmg_to_simple(g)
    # g_relabeled = nx.convert_node_labels_to_integers(g_lmg)
    # g_pyn = simple_to_nauty(g_relabeled)
    # order = pn.canon_label(g_pyn)
    # print(order)

    order = canon_node_order(rule.graph)
    print(order)
