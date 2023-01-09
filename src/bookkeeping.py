import random
from typing import Collection

from cnrg.Rule import MetaRule
from cnrg.VRG import VRG


# TODO: move ancestor() and common_ancestor() to VRG.py?
def ancestor(u: int, grammar: VRG, t: int) -> tuple[int, MetaRule, str]:
    parent_idx = grammar.cover[t][u]  # points to which entry in rule_tree contains this rule
    parent_metarule = grammar.decomposition[parent_idx][0]  # the rule in question
    ancestor_u = parent_metarule[t].alias[u]  # the node in the rule's RHS corresponding to u
    return parent_idx, parent_metarule, ancestor_u


def common_ancestor(nodes: Collection[int], grammar: VRG, t: int,) -> tuple[int, MetaRule, dict[int, str]]:
    if len(nodes) == 1:
        u, = nodes
        parent_idx, parent_metarule, ancestor_u = ancestor(u, grammar, t=t)
        return parent_idx, parent_metarule, {u: ancestor_u}

    parent_indices: dict[int, int] = {u: grammar.cover[t][u] for u in nodes}
    node_ancestors: dict[int, list[int]] = {u: [] for u in nodes}

    # trace the ancestral lineage of each node all the way up to the root
    for u in nodes:
        while parent_indices[u] is not None:
            node_ancestors[u] += [parent_indices[u]]
            parent_indices[u] = grammar.decomposition[parent_indices[u]][1]

    # take the rule furthest away from the root in the decomposition that still covers everyone
    common_ancestors = set.intersection(*[set(lineage) for lineage in node_ancestors.values()])
    least_common_idx = max(common_ancestors, key=grammar.level)  # take the least (i.e., furthest from the root) common ancestor

    ancestor_metarule = grammar.decomposition[least_common_idx][0]

    ancestor_nodes = {}  # type: ignore
    for u in nodes:
        if len(node_ancestors[u]) == 0:
            ancestor_nodes[u] = ancestor_metarule[t].alias[u]
        else:
            pre_ancestor_idx = node_ancestors[u][node_ancestors[u].index(least_common_idx) - 1]
            ancestor_u = grammar.decomposition[pre_ancestor_idx][2]
            ancestor_nodes[u] = ancestor_u if ancestor_u is not None else ancestor_metarule[t].alias[u]

    return least_common_idx, ancestor_metarule, ancestor_nodes


def propagate_ancestors(node: str, rule_idx: int, grammar: VRG, t1: int, t2: int, mode: str, stop_at: int = -1):
    assert mode in ('add', 'del')
    if (node is None and rule_idx is None) or stop_at == rule_idx:
        return
    if node is None or rule_idx is None:
        raise AssertionError('decomposition\'s miffed')

    metarule, pidx, anode = grammar.decomposition[rule_idx]
    if mode == 'add':
        metarule[t2].lhs = max(1, metarule[t2].lhs + 1)
        metarule[t2].graph.nodes[node]['b_deg'] += 1
        metarule[t2].graph.nodes[node]['label'] += 1
    else:
        metarule[t2].lhs -= 1
        metarule[t2].graph.nodes[node]['b_deg'] -= 1
        metarule[t2].graph.nodes[node]['label'] -= 1

    propagate_ancestors(anode, pidx, grammar, t1, t2, mode, stop_at=stop_at)


def propagate_descendants(nts: str, rule_idx: int, grammar: VRG, t1: int, t2: int, mode: str = 'add'):
    assert mode in ('add', 'del')

    for child_idx, child_metarule in grammar.find_children_of(nts, rule_idx, t2):
        # child_metarule.ensure(t1, t2)

        if mode == 'add':
            (v, d), = random.sample(child_metarule[t2].graph.nodes(data=True), 1)
            d['b_deg'] += 1
            child_metarule[t2].lhs += 1

            if 'label' in d:
                d['label'] += 1
                propagate_descendants(v, child_idx, grammar, t1, t2, mode=mode)
        else:
            assert sum(d['b_deg'] for v, d in child_metarule[t2].graph.nodes(data=True)) > 0

            (v, d), = random.sample(child_metarule[t2].graph.nodes(data=True), 1)
            while d['b_deg'] == 0:
                (v, d), = random.sample(child_metarule[t2].graph.nodes(data=True), 1)

            d['b_deg'] -= 1
            child_metarule[t2].lhs -= 1

            if 'label' in d:
                d['label'] -= 1
                propagate_descendants(v, child_idx, grammar, t1, t2, mode=mode)
