import sys
from typing import Collection

# from tqdm import tqdm
import networkx as nx

sys.path.append('..')

from utils import is_rule_isomorphic

from cnrg.Rule import PartRule
from cnrg.VRG import VRG
from cnrg.Tree import create_tree
from cnrg.LightMultiGraph import convert
from cnrg.LightMultiGraph import LightMultiGraph as LMG
from cnrg.extract import MuExtractor
from cnrg.partitions import leiden, louvain


def create_splitting_rule(subgrammars: list[VRG], time: int) -> PartRule:
    S = min(min(key for key in subgrammar.rule_dict) for subgrammar in subgrammars)
    rhs = nx.Graph()

    for idx, subgrammar in enumerate(subgrammars):
        rhs.add_node(str(idx), b_deg=0, label=min(key for key in subgrammar.rule_dict))

    return PartRule(S - 1, rhs, time=time)


def merge_subgrammars(splitting_rule: PartRule, subgrammars: list[VRG]) -> VRG:
    for subgrammar in subgrammars:
        subgrammar.push_down_grammar()

    for i, subgrammar in enumerate(subgrammars):
        if i == 0:
            supergrammar = subgrammar
            splitting_idx = len(supergrammar.rule_tree)

            # make the root of the old decomposition point to the new root
            for idx, (_, parent_idx, ancestor_node) in enumerate(supergrammar.rule_tree):
                if parent_idx is None and ancestor_node is None:
                    supergrammar.rule_tree[idx][1] = splitting_idx
                    supergrammar.rule_tree[idx][2] = str(i)
                    break
            else:
                raise AssertionError('never find the root rule')

            supergrammar.rule_tree += [[splitting_rule, None, None]]
            supergrammar.rule_list += [splitting_rule]
            supergrammar.rule_dict[splitting_rule.lhs] = [splitting_rule]
        else:
            offset = len(supergrammar.rule_tree)

            # merge in new rules that are duplicates of old rules
            subtree = subgrammar.rule_tree.copy()
            for idx, (subrule, _, _) in enumerate(subtree):
                if not supergrammar.find_rule(subrule, where='rule_list'):
                    for superrule in supergrammar.rule_dict[subrule.lhs]:
                        if f := is_rule_isomorphic(subrule, superrule):
                            superrule.frequency += subrule.frequency
                            superrule.subtree |= subrule.subtree

                            for u in subrule.mapping:
                                superrule.mapping[u] = f[subrule.mapping[u]]

                            subgrammar.replace_rule(subrule, superrule, f)

                            break
                    else:
                        supergrammar.rule_list += [subrule]

                        if subrule.lhs in supergrammar.rule_dict:
                            supergrammar.rule_dict[subrule.lhs] += [subrule]
                        else:
                            supergrammar.rule_dict[subrule.lhs] = [subrule]

            # shift the indices of the sub-decomposition
            for idx, (_, parent_idx, ancestor_node) in enumerate(subgrammar.rule_tree):
                if parent_idx is None and ancestor_node is None:
                    subgrammar.rule_tree[idx][1] = splitting_idx
                    subgrammar.rule_tree[idx][2] = str(i)
                else:
                    subgrammar.rule_tree[idx][1] += offset

            # shift the indices of the covering_idx map
            for node in subgrammar.covering_idx:
                subgrammar.covering_idx[node] += offset

            # append the sub-decomposition to the super-decomposition
            supergrammar.rule_tree += subgrammar.rule_tree

            assert len(supergrammar.covering_idx.keys() & subgrammar.covering_idx.keys()) == 0
            supergrammar.covering_idx |= subgrammar.covering_idx

    # # recompute the rule_dict
    # supergrammar.rule_dict = {}
    # for rule in supergrammar.rule_list:
    #     if rule.lhs in supergrammar.rule_dict.keys():
    #         supergrammar.rule_dict[rule.lhs] += [rule]
    #     else:
    #         supergrammar.rule_dict[rule.lhs] = [rule]

    # # recompute grammar references in rule_tree
    # for tree_idx, (rule, _, _) in enumerate(supergrammar.rule_tree):
    #     try:
    #         list_idx = supergrammar.rule_list.index(rule)
    #         supergrammar.rule_tree[tree_idx][0] = supergrammar.rule_list[list_idx]
    #     except IndexError as e:
    #         print(tree_idx, rule)
    #         raise IndexError from e

    return supergrammar


def decompose(g: nx.Graph, time: int = 0, mu: int = 4, clustering: str = 'leiden', gtype: str = 'mu_level_dl', name: str = ''):
    if g.order() == 0:
        raise AssertionError('!!! graph is empty !!!')

    if float(nx.__version__[:3]) < 2.4:
        connected_components = nx.connected_component_subgraphs(g)
    else:
        connected_components = [g.subgraph(comp) for comp in nx.connected_components(g)]

    if len(connected_components) == 1:
        supergrammar = decompose_component(g, clustering=clustering, gtype=gtype, name=name, mu=mu)
    else:
        subgrammars = [decompose_component(component, clustering=clustering, gtype=gtype, name=name, mu=mu)
                       for component in connected_components]

        splitting_rule = create_splitting_rule(subgrammars, time)
        supergrammar = merge_subgrammars(splitting_rule, subgrammars)

    # sanity check
    for v in g.nodes():
        assert v in supergrammar.covering_idx

    # proper bookkeeping
    for idx, rule in enumerate(supergrammar.rule_list):
        rule.idn = idx
        rule.time = time
        rule.time_changed = time

    return supergrammar


def decompose_component(g: nx.Graph, mu: int = 4, clustering: str = 'leiden', gtype: str = 'mu_level_dl', name: str = ''):
    if not isinstance(g, LMG):
        g = convert(g)

    if clustering == 'leiden':
        clusters = leiden(g)
    elif clustering == 'louvain':
        clusters = louvain(g)
    else:
        raise NotImplementedError

    dendrogram = create_tree(clusters)

    vrg = VRG(clustering=clustering,
              gtype=gtype,
              name=name,
              mu=mu)

    extractor = MuExtractor(g=g.copy(),
                            gtype=gtype,
                            grammar=vrg,
                            mu=mu,
                            root=dendrogram)

    extractor.generate_grammar()
    # ex_sequence = extractor.extracted_sequence
    grammar = extractor.grammar

    # for rule in grammar.rule_list:
    #     rule.time = time

    return grammar


def ancestor(u: int, grammar: VRG) -> tuple[PartRule, int, str]:
    parent_idx = grammar.covering_idx[u]  # points to which entry in rule_tree contains this rule
    parent_rule = grammar.rule_tree[parent_idx][0]  # the rule in question
    ancestor_u = parent_rule.mapping[u]  # the node in the rule's RHS corresponding to u
    return parent_rule, parent_idx, ancestor_u


def common_ancestor(nodes: Collection[int], grammar: VRG) -> tuple[PartRule, int, dict[int, str]]:
    if len(nodes) == 1:
        u, = nodes
        parent_rule, parent_idx, ancestor_u = ancestor(u, grammar)
        return parent_rule, parent_idx, {u: ancestor_u}

    parent_indices: dict[int, int] = {u: grammar.covering_idx[u] for u in nodes}
    node_ancestors: dict[int, list[int]] = {u: [] for u in nodes}

    # trace the ancestral lineage of each node all the way up to the root
    for u in nodes:
        while parent_indices[u] is not None:
            try:
                node_ancestors[u] += [parent_indices[u]]
                parent_indices[u] = grammar.rule_tree[parent_indices[u]][1]
            except IndexError as E:
                print(parent_indices[u], len(grammar.rule_tree))
                raise IndexError from E

    # take the rule furthest away from the root in the decomposition that still covers everyone
    common_ancestors = set.intersection(*[set(lineage) for lineage in node_ancestors.values()])
    least_common_ancestor = max(common_ancestors, key=lambda idx: grammar.find_level(grammar.rule_tree[idx][0]))
    # least_common_ancestor = max(common_ancestors, key=lambda idx: grammar.rule_tree[idx][0].level)

    ancestor_idx = least_common_ancestor
    ancestor_rule = grammar.rule_tree[ancestor_idx][0]

    ancestor_nodes = {}  # type: ignore
    for u in nodes:
        if len(node_ancestors[u]) == 0:
            ancestor_nodes[u] = ancestor_rule.mapping[u]
            # ancestor_nodes[u] = grammar.which_rule_source[u]
        else:
            pre_ancestor_idx = node_ancestors[u][node_ancestors[u].index(least_common_ancestor) - 1]
            ancestor_u = grammar.rule_tree[pre_ancestor_idx][2]
            ancestor_nodes[u] = ancestor_u if ancestor_u is not None else ancestor_rule.mapping[u]

    return ancestor_rule, ancestor_idx, ancestor_nodes
