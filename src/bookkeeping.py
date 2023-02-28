import random
from typing import Collection

from dyverg.Rule import MetaRule
from dyverg.VRG import VRG


# TODO: move ancestor() and common_ancestor() to VRG.py?
def ancestor(u: int, grammar: VRG) -> tuple[int, MetaRule, str]:
    for t in grammar.times[::-1]:
        if u in grammar.cover[t]:
            parent_idx = grammar.cover[t][u]  # points to which entry in rule_tree contains this rule
            break
    else:
        raise AssertionError(f'!!! {u} is not covered by the grammar !!!')
    parent_metarule = grammar.decomposition[parent_idx][0]  # the rule in question
    ancestor_u = parent_metarule.alias[u]  # the node in the rule's RHS corresponding to u
    return parent_idx, parent_metarule, ancestor_u


def common_ancestor(nodes: Collection[int], grammar: VRG) -> tuple[int, MetaRule, dict[int, str]]:
    if len(nodes) == 1:
        u, = nodes
        parent_idx, parent_metarule, ancestor_u = ancestor(u, grammar)
        return parent_idx, parent_metarule, {u: ancestor_u}

    # parent_indices: dict[int, int] = {u: grammar.cover[t][u] for u in nodes}
    # parent_indices: dict[int, int] = {u: ancestor(u, grammar)[0]}
    lineages: dict[int, list[int]] = {u: [] for u in nodes}

    # trace the ancestral lineage of each node all the way up to the root
    for u in nodes:
        uparent, _, _ = ancestor(u, grammar)
        while uparent is not None:
            lineages[u].append(uparent)
            uparent = grammar.decomposition[uparent][1]
            # parent_indices[u] = grammar.decomposition[parent_indices[u]][1]

    # take the rule furthest away from the root in the decomposition that still covers everyone
    common_ancestors = set.intersection(*[set(lineage) for lineage in lineages.values()])
    least_common_idx = max(common_ancestors, key=grammar.level)  # take the least (i.e., furthest from the root) common ancestor

    ancestor_metarule = grammar.decomposition[least_common_idx][0]

    ancestor_nodes = {}  # type: ignore
    for u in nodes:
        if len(lineages[u]) == 0:
            ancestor_nodes[u] = ancestor_metarule.alias[u]
        else:
            pre_ancestor_idx = lineages[u][lineages[u].index(least_common_idx) - 1]
            ancestor_u = grammar.decomposition[pre_ancestor_idx][2]
            ancestor_nodes[u] = ancestor_u if ancestor_u is not None else ancestor_metarule.alias[u]

    return least_common_idx, ancestor_metarule, ancestor_nodes


# removes nonterminals referring to this rule up the decomposition
def redact(grammar: VRG, idx: int, nts: int, time: int):
    if not idx:
        return

    metarule, pidx, anode = grammar.decomposition[idx]

    assert 'label' in metarule[time].graph.nodes[nts]
    metarule[time].graph.remove_node(nts)

    if metarule[time].graph.order() == 0:
        redact(grammar, pidx, anode, time)


# reintroduces nonterminals referring to this rule up the decomposition
def unseal(grammar: VRG, idx: int, nts: int, label: int, time: int):
    if not idx:
        return

    metarule, pidx, anode = grammar.decomposition[idx]

    # if metarule[time].graph.order() == 0:
    #     unseal(grammar, pidx, anode, time)
    unseal(grammar, pidx, anode, metarule[time].lhs, time)

    # assert nts not in metarule[time].graph
    if nts not in metarule[time].graph:
        metarule[time].graph.add_node(nts, b_deg=label, label=label)
    # if nts not in grammar[idx][time].graph:
    #     grammar[idx][time].graph.add_node(nts, b_deg=0, label=0)


def propagate_ancestors(nts: str, rule_idx: int, child_lhs: int, grammar: VRG,
                        t1: int, t2: int, mode: str, stop_at: int = -1):
    assert mode in ('add', 'del')
    if (nts is None and rule_idx is None) or stop_at == rule_idx:
        return
    if nts is None or rule_idx is None:
        raise AssertionError('decomposition\'s miffed')

    metarule, pidx, anode = grammar.decomposition[rule_idx]

    if nts not in metarule[t2].graph:
        # import pdb
        # pdb.set_trace()
        metarule[t2].graph.add_node(nts, b_deg=child_lhs, label=child_lhs)
    else:
        metarule[t2].graph.nodes[nts]['label'] = child_lhs

    if mode == 'add':
        # metarule[t2].lhs += 1
        metarule[t2].lhs = max(1, metarule[t2].lhs + 1)
        # metarule[t2].graph.nodes[nts]['label'] += 1
        # metarule[t2].graph.nodes[nts]['label'] = max(1, metarule[t2].graph.nodes[nts]['label'] + 1)
        metarule[t2].graph.nodes[nts]['b_deg'] += 1
    else:
        # metarule[t2].lhs -= 1
        metarule[t2].lhs = max(0, metarule[t2].lhs - 1)
        # metarule[t2].graph.nodes[nts]['label'] -= 1
        # metarule[t2].graph.nodes[nts]['label'] = max(0, metarule[t2].graph.nodes[nts]['label'] - 1)
        metarule[t2].graph.nodes[nts]['b_deg'] -= 1

        if metarule[t2].graph.nodes[nts]['b_deg'] < 0:
            import pdb
            pdb.set_trace()
        assert metarule[t2].graph.nodes[nts]['b_deg'] >= 0

    propagate_ancestors(anode, pidx, metarule[t2].lhs, grammar, t1, t2, mode, stop_at=stop_at)


def propagate_descendants(nts: str, rule_idx: int, grammar: VRG, t1: int, t2: int, mode: str = 'add'):
    assert mode in ('add', 'del')

    for child_idx, child_metarule in grammar.find_children_of(nts, rule_idx, t2):
        # child_metarule.ensure(t1, t2)

        if mode == 'add':
            (v, d), = random.sample(child_metarule[t2].graph.nodes(data=True), 1)
            # child_metarule[t2].lhs += 1
            child_metarule[t2].lhs = max(1, child_metarule[t2].lhs + 1)
            d['b_deg'] += 1

            if 'label' in d:
                # d['label'] += 1
                d['label'] = max(1, d['label'] + 1)
                propagate_descendants(v, child_idx, grammar, t1, t2, mode=mode)
        else:
            assert sum(d['b_deg'] for v, d in child_metarule[t2].graph.nodes(data=True)) > 0

            (v, d), = random.sample(child_metarule[t2].graph.nodes(data=True), 1)
            while d['b_deg'] == 0:
                (v, d), = random.sample(child_metarule[t2].graph.nodes(data=True), 1)

            # child_metarule[t2].lhs -= 1
            child_metarule[t2].lhs = max(0, child_metarule[t2].lhs - 1)
            d['b_deg'] -= 1

            if 'label' in d:
                # d['label'] -= 1
                d['label'] = max(0, d['label'] - 1)
                propagate_descendants(v, child_idx, grammar, t1, t2, mode=mode)
