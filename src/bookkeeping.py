import sys
from typing import Collection

# from tqdm import tqdm
import numpy as np
import networkx as nx

sys.path.append('..')

# from utils import silence
from cnrg.Rule import PartRule
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


def decompose(g: nx.Graph, time: int = 0, mu: int = 4,
              clustering: str = 'leiden', gtype: str = 'mu_level_dl',
              name: str = ''):
    if g.order() == 0:
        raise AssertionError('!!! graph is empty !!!')

    if float(nx.__version__[:3]) < 2.4:
        connected_components = nx.connected_component_subgraphs(g)
    else:
        connected_components = [g.subgraph(comp) for comp in nx.connected_components(g)]

    # connected_components = [g for g in connected_components if g.order() > mu]
    if len(connected_components) == 1:
        supergrammar = decompose_component(g, time=time)
    else:
        subgrammars = [decompose_component(component, time=time, clustering=clustering, gtype=gtype, name=name, mu=mu)
                       for component in connected_components]

        S = min(min(key for key in subgrammar.rule_dict) for subgrammar in subgrammars) - 1
        rhs = nx.Graph()
        rhs.add_nodes_from(map(chr, range(len(subgrammars))), b_deg=0, label=S)
        root = PartRule(S, rhs)

        for i, subgrammar in enumerate(subgrammars):
            if i == 0:
                supergrammar = subgrammar

                # shift the indices of the decomposition
                for idx, _ in enumerate(supergrammar.rule_tree):
                    if supergrammar.rule_tree[idx][1] is not None:
                        supergrammar.rule_tree[idx][1] += 1
                    else:
                        supergrammar.rule_tree[idx][1] = 0
                        supergrammar.rule_tree[idx][2] = list(root.graph.nodes()).index(chr(i))

                # shift the indices of the rule_source map
                for idx in supergrammar.rule_source:
                    supergrammar.rule_source[idx] += 1

                # append the rule tree, so that it is a branch under the home decomposition
                supergrammar.rule_tree = [[root, None, None]] + supergrammar.rule_tree

                # incorporate the new root rule
                supergrammar.rule_list += [root]
                supergrammar.rule_dict[S] = [root]
            else:
                offset = len(supergrammar.rule_tree)

                # shift the indices of the decomposition
                for idx, _ in enumerate(subgrammar.rule_tree):
                    if subgrammar.rule_tree[idx][1] is not None:
                        subgrammar.rule_tree[idx][1] += offset
                    else:
                        subgrammar.rule_tree[idx][1] = 0
                        subgrammar.rule_tree[idx][2] = list(root.graph.nodes()).index(chr(i))

                # shift the indices of the rule_source map
                for idx in subgrammar.rule_source:
                    subgrammar.rule_source[idx] += offset

                # APPEND the rule tree, so that it is a branch under the home decomposition
                # if we were to PREPEND instead, the common_ancestor(...) would no longer work
                supergrammar.rule_tree += subgrammar.rule_tree

                # merge in new rules that are duplicates of old rules
                for subrule in subgrammar.rule_list:
                    try:
                        found_idx = supergrammar.rule_list.index(subrule)
                        supergrammar.rule_list[found_idx].frequency += 1
                    except ValueError:
                        supergrammar.num_rules += 1
                        supergrammar.rule_list += [subrule]

                        if subrule.lhs in supergrammar.rule_dict:
                            supergrammar.rule_dict[subrule.lhs] += [subrule]
                        else:
                            supergrammar.rule_dict[subrule.lhs] = [subrule]

                # merge the bookkeeping dicts
                # the node sets should be disjoint, so this is fine
                supergrammar.rule_source |= subgrammar.rule_source
                supergrammar.which_rule_source |= subgrammar.which_rule_source

        # recompute the rule_dict
        supergrammar.rule_dict = {}
        for rule in supergrammar.rule_list:
            if rule.lhs in supergrammar.rule_dict.keys():
                supergrammar.rule_dict[rule.lhs] += [rule]
            else:
                supergrammar.rule_dict[rule.lhs] = [rule]

        # recompute grammar references in rule_tree
        for tree_idx, (rule, _, _) in enumerate(supergrammar.rule_tree):
            try:
                list_idx = supergrammar.rule_list.index(rule)
                supergrammar.rule_tree[tree_idx][0] = supergrammar.rule_list[list_idx]
            except IndexError as e:
                print(tree_idx, rule)
                raise IndexError from e

        for v in g.nodes():
            for subgrammar in subgrammars:
                if v in subgrammar.rule_source:
                    break
            else:
                print(v, end=', ')
            # for 
        print()

        for v in g.nodes():
            assert v in supergrammar.rule_source

    return supergrammar


def decompose_component(g: nx.Graph, time: int = 0, mu: int = 4,
                        clustering: str = 'leiden', gtype: str = 'mu_level_dl',
                        name: str = ''):
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

    for rule in grammar.rule_list:
        rule.time = time

    return grammar


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
