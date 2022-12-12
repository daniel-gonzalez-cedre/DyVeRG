import random
from typing import Collection, Iterable
from functools import reduce

import networkx as nx

from cnrg.Rule import PartRule
from cnrg.VRG import VRG
from cnrg.Tree import create_tree
from cnrg.LightMultiGraph import convert
from cnrg.LightMultiGraph import LightMultiGraph as LMG
from cnrg.extract import MuExtractor
from cnrg.partitions import leiden, louvain

from src.utils import is_rule_isomorphic


# returns the isomorphic rule after merging if one is found, otherwise returns the new rule
def assimilate_rule(new_rule: PartRule, grammar: VRG, parallel: bool = True, compute_ged: bool = False) -> tuple[PartRule, dict[str, str]]:
    return NotImplemented
    if new_rule.lhs in grammar.rule_dict:
        found = False
        for old_rule in grammar.rule_dict[new_rule.lhs]:
            if new_rule is not old_rule:
                if (f := is_rule_isomorphic(new_rule, old_rule)):
                    for x in new_rule.mapping:
                        old_rule.mapping[x] = f[new_rule.mapping[x]]

                    old_rule.edit_dist = min(old_rule.edit_dist, new_rule.edit_dist)  # TODO: think about this
                    old_rule.frequency += new_rule.frequency
                    old_rule.subtree |= new_rule.subtree

                    grammar.replace_rule(new_rule, old_rule, f)
                    return old_rule, f
            else:
                found = True
                # maybe add a break here? idk
        if not found:
            grammar.rule_dict[new_rule.lhs] += [new_rule]
            grammar.rule_list += [new_rule]
    else:
        grammar.rule_dict[new_rule.lhs] = [new_rule]
        grammar.rule_list += [new_rule]

    # only computed when joining two grammars, not for independent updates
    if compute_ged:
        new_rule.edit_dist = grammar.minimum_edit_dist(new_rule, parallel=parallel)

    return new_rule, {v: v for v in new_rule.graph.nodes()}


def assimilate_rules(host_grammar: VRG, parasite_grammar: VRG, parallel: bool = True):
    return NotImplemented
    for parasite_rule in parasite_grammar.rule_list:
        assimilated_rule, f = assimilate_rule(parasite_rule, host_grammar, parallel=parallel, compute_ged=True)
        parasite_grammar.replace_rule(parasite_rule, assimilated_rule, f)


def ancestor(u: int, grammar: VRG) -> tuple[int, PartRule, str]:
    parent_idx = grammar.cover[u]  # points to which entry in rule_tree contains this rule
    parent_rule = grammar[parent_idx][0]  # the rule in question
    ancestor_u = parent_rule.alias[u]  # the node in the rule's RHS corresponding to u
    return parent_idx, parent_rule, ancestor_u


def common_ancestor(nodes: Collection[int], grammar: VRG) -> tuple[int, PartRule, dict[int, str]]:
    if len(nodes) == 1:
        u, = nodes
        parent_rule, parent_idx, ancestor_u = ancestor(u, grammar)
        return parent_rule, parent_idx, {u: ancestor_u}

    parent_indices: dict[int, int] = {u: grammar.cover[u] for u in nodes}
    node_ancestors: dict[int, list[int]] = {u: [] for u in nodes}

    # trace the ancestral lineage of each node all the way up to the root
    for u in nodes:
        while parent_indices[u] is not None:
            node_ancestors[u] += [parent_indices[u]]
            parent_indices[u] = grammar[parent_indices[u]][1]

    # take the rule furthest away from the root in the decomposition that still covers everyone
    common_ancestors = set.intersection(*[set(lineage) for lineage in node_ancestors.values()])
    least_common_ancestor = max(common_ancestors, key=grammar.level)

    ancestor_idx = least_common_ancestor
    # ancestor_rule = grammar.decomposition[ancestor_idx][0]
    ancestor_rule = grammar[ancestor_idx][0]

    ancestor_nodes = {}  # type: ignore
    for u in nodes:
        if len(node_ancestors[u]) == 0:
            ancestor_nodes[u] = ancestor_rule.alias[u]
        else:
            pre_ancestor_idx = node_ancestors[u][node_ancestors[u].index(least_common_ancestor) - 1]
            ancestor_u = grammar[pre_ancestor_idx][2]
            ancestor_nodes[u] = ancestor_u if ancestor_u is not None else ancestor_rule.alias[u]

    return ancestor_idx, ancestor_rule, ancestor_nodes


def propagate_ancestors(node: str, rule_idx: int, grammar: VRG, time: int = None,
                        edit: bool = False, stop_at: int = -1):
    if (node is None and rule_idx is None) or stop_at == rule_idx:
        return
    if node is None or rule_idx is None:
        raise AssertionError('decomposition\'s miffed')

    rule, pidx, anode = grammar[rule_idx]

    rule.lhs = rule.lhs + 1 if rule.lhs >= 0 else 1
    rule.time_changed = time
    rule.graph.nodes[node]['b_deg'] += 1
    rule.graph.nodes[node]['label'] += 1

    if edit:
        # modified_rule.edit_dist += 2  # TODO: think about this
        rule.edit_dist += 1

    propagate_ancestors(anode, pidx, grammar, time, stop_at=stop_at)


def propagate_descendants(nts: str, rule_idx: int, grammar: VRG, time: int = None):
    for child_idx, child_rule in grammar.find_children_of(nts, rule_idx):
        (v, d), = random.sample(child_rule.graph.nodes(data=True), 1)
        d['b_deg'] += 1

        if 'label' in d:
            d['label'] += 1
            child_rule.edit_dist += 1  # cost of relabeling a node

            propagate_descendants(v, child_idx, grammar, time)

        child_rule.lhs += 1
        child_rule.edit_dist += 2  # cost of relabeling a node and changing the RHS

        if time:
            child_rule.time_changed = time

        # assimilate_rule(child_rule, grammar)


def create_splitting_rule(subgrammars: Iterable[VRG], time: int) -> PartRule:
    rhs = nx.Graph()
    S = min(min(rule.lhs for rule, _, _ in subgrammar.decomposition)
            for subgrammar in subgrammars)

    for idx, subgrammar in enumerate(subgrammars):
        rhs.add_node(str(idx), b_deg=0, label=subgrammar.root_rule.lhs)

    # S = sum(d['b_deg'] for v, d in rhs.nodes(data=True) if 'label' in d)
    return PartRule(S - 1, rhs, time=time)


def decompose(g: nx.Graph, time: int = 0, mu: int = 4, clustering: str = 'leiden', gtype: str = 'mu_level_dl', name: str = '', verbose: bool = False):
    def merge_subgrammars(splitting_rule: PartRule, subgrammars: list[VRG]) -> VRG:
        # for subgrammar in subgrammars:
        #     subgrammar.push_down_grammar()

        supergrammar = subgrammars[0]
        splitting_rule.idn = len(supergrammar.decomposition)
        # splitting_idx = len(supergrammar.decomposition)

        # make the root of the old decomposition point to the new root
        for idx, (_, pidx, anode) in enumerate(supergrammar.decomposition):
            if pidx is None and anode is None:
                supergrammar[idx][1] = splitting_rule.idn
                supergrammar[idx][2] = '0'
                break
        else:
            raise AssertionError('never found the root rule')

        supergrammar.decomposition += [[splitting_rule, None, None]]
        # supergrammar.rule_list += [splitting_rule]
        # supergrammar.rule_dict[splitting_rule.lhs] = [splitting_rule]

        for i, subgrammar in enumerate(subgrammars[1:], start=1):
            offset = len(supergrammar.decomposition)

            # merge in new rules that are duplicates of old rules
            # assimilate_rules(supergrammar, subgrammar)

            # shift the indices of the sub-decomposition
            for idx, (_, pidx, anode) in enumerate(subgrammar.decomposition):
                subgrammar[idx][0].idn += offset
                if pidx is None and anode is None:
                    subgrammar[idx][1] = splitting_rule.idn
                    subgrammar[idx][2] = str(i)
                else:
                    subgrammar[idx][1] += offset

            # append the sub-decomposition to the super-decomposition
            supergrammar.decomposition += subgrammar.decomposition

            # shift the indices of the covering_idx map
            for node in subgrammar.cover:
                subgrammar.cover[node] += offset

            assert len(supergrammar.cover.keys() & subgrammar.cover.keys()) == 0
            supergrammar.cover |= subgrammar.cover

        return supergrammar

    if g.order() == 0:
        raise AssertionError('!!! graph is empty !!!')

    if float(nx.__version__[:3]) < 2.4:
        connected_components = nx.connected_component_subgraphs(g)
    else:
        connected_components = [g.subgraph(comp) for comp in nx.connected_components(g)]

    if len(connected_components) == 1:
        supergrammar = decompose_component(g, clustering=clustering, gtype=gtype, name=name, mu=mu, verbose=verbose)
    else:
        subgrammars = [decompose_component(component, clustering=clustering, gtype=gtype, name=name, mu=mu, verbose=verbose)
                       for component in connected_components]

        splitting_rule = create_splitting_rule(subgrammars, time)
        supergrammar = merge_subgrammars(splitting_rule, subgrammars)

    # sanity check
    for v in g.nodes():
        assert v in supergrammar.cover

    supergrammar.compute_rules()
    supergrammar.compute_levels()
    supergrammar.set_time(time)

    return supergrammar


def decompose_component(g: nx.Graph, mu: int = 4, clustering: str = 'leiden', gtype: str = 'mu_level_dl', name: str = '', verbose: bool = False):
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

    extractor.generate_grammar(verbose=verbose)
    grammar = extractor.grammar

    return grammar
