import sys
from typing import Collection

import numpy as np
import networkx as nx

sys.path.append('..')

from cnrg.VRG import VRG
from cnrg.Tree import create_tree
from cnrg.LightMultiGraph import LightMultiGraph as LMG
from cnrg.extract import MuExtractor
from cnrg.partitions import leiden, louvain


def convert_LMG(g: nx.Graph):
    g_lmg = LMG()
    g_lmg.add_nodes_from(g.nodes())
    g_lmg.add_edges_from(g.edges())
    return g_lmg


def decompose(g: nx.Graph, clustering: str = 'leiden', gtype: str = 'mu_level_dl', name: str = '', mu: int = 4):
    if not isinstance(g, LMG):
        g = convert_LMG(g)

    if clustering == 'leiden':
        clusters = leiden(g)
    elif clustering == 'louvain':
        clusters = louvain(g)
    else:
        raise NotImplementedError

    dendrogram = create_tree(clusters)

    vrg = VRG(clustering=clustering,
              type=gtype,
              name=name,
              mu=mu)

    extractor = MuExtractor(g=g.copy(),
                            type=gtype,
                            grammar=vrg,
                            mu=mu,
                            root=dendrogram)

    extractor.generate_grammar()
    # ex_sequence = extractor.extracted_sequence
    grammar = extractor.grammar

    return grammar
    # return extractor, grammar


def ancestor(u: int, grammar: VRG):
    which_u = grammar.which_rule_source[u]  # points to which node in the rule's RHS corresponds to u
    which_parent = grammar.rule_source[u]  # points to which entry in rule_tree contains this rule
    parent_rule = grammar.rule_tree[which_parent][0]  # the rule in question

    return parent_rule, which_parent, which_u


def common_ancestor(nodes: Collection[int], grammar: VRG):
    if len(nodes) == 1:
        u, = nodes
        parent_idx = grammar.rule_source[u]
        parent_rule = grammar.rule_tree[parent_idx][0]
        which_children = {u: grammar.which_rule_source[u]}
        return parent_rule, parent_idx, which_children

    indices: dict[int, int] = {u: grammar.rule_source[u] for u in nodes}
    ancestors: dict[int, list[int]] = {u: [] for u in nodes}

    # trace the ancestral lineage of u all the way to the root
    for u in nodes:
        while indices[u] is not None:
            try:
                ancestors[u] += [indices[u]]
                indices[u] = grammar.rule_tree[indices[u]][1]
            except IndexError:
                print(indices[u])
                print(len(grammar.rule_tree))
                exit()
                pass

    common_ancestors = set.intersection(*[set(lineage) for lineage in ancestors.values()])
    common_ancestor = min(common_ancestors)

    parent_idx = common_ancestor
    parent_rule = grammar.rule_tree[parent_idx][0]

    which_children = {}  # type: ignore
    for u in nodes:
        if len(ancestors[u]) == 0:
            which_children[u] = grammar.which_rule_source[u]
        else:
            u_child = ancestors[u][ancestors[u].index(common_ancestor) - 1]
            which_child_u = grammar.rule_tree[u_child][2]
            which_children[u] = which_child_u if which_child_u is not None else grammar.which_rule_source[u]
            # print(grammar.rule_tree[which_children[u]])

            # which_children[u] = grammar.rule_tree[u_child][2]
            # if which_children[u] is None:
            #     print(u_child)
            #     print(grammar.rule_tree[u_child])

    # for x, y in which_children.items():
    #     if y is None:
    #         print('nodes', nodes)
    #         print('lens', [len(ancestors[u]) for u in which_children])
    #         print('which_children', which_children)
    #         print('parent_rule', parent_rule)
    #         print('parent_rule.graph', parent_rule.graph.nodes())
    #         print('parent_idx', parent_idx)

    return parent_rule, parent_idx, which_children


def deprecated_common_ancestor(u: int, v: int, grammar: VRG):
    ind_u = grammar.rule_source[u]
    ind_v = grammar.rule_source[v]

    u_anc = []
    v_anc = []

    # trace the ancestral lineage of u all the way to the root
    while ind_u is not None:
        u_anc.append(ind_u)
        ind_u = grammar.rule_tree[ind_u][1]

    # trace the ancestral lineage of v until it intersects u's lineage
    while ind_v not in u_anc:
        v_anc.append(ind_v)
        ind_v = grammar.rule_tree[ind_v][1]

        # somehow the two paths did not cross
        if ind_v is None:
            return np.log(0)

    parent_idx = ind_v

    parent_rule = grammar.rule_tree[parent_idx][0]
    which_child_u = None
    which_child_v = None

    if len(v_anc) > 0:
        which_child_v = grammar.rule_tree[v_anc[-1]][2]
    else:
        which_child_v = grammar.which_rule_source[v]

    if u_anc.index(parent_idx) == 0:
        which_child_u = grammar.which_rule_source[u]
    else:
        which_child_u = grammar.rule_tree[u_anc[u_anc.index(parent_idx) - 1]][2]

    return parent_rule, parent_idx, which_child_u, which_child_v
