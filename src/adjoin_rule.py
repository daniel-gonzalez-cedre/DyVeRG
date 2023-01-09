from networkx.exception import NetworkXError

from cnrg.VRG import VRG
from src.bookkeeping import ancestor, common_ancestor, propagate_ancestors, propagate_descendants


def domestic(grammar: VRG, u: int, v: int, t1: int, t2: int, mode: str):  # TODO: update docstring instructions
    """
        Given an edge event (u, v), where both u and v are already known to the grammar G:
            1. Find the lowest rule R that covers both u and v simultaneously
            2. Update R -> R' by adding or deleting an edge between u and v on the RHS of R
            3. Merge the modified R' back into G
            4. If R was learned at time t: remove R from G; otherwise, preserve R

        Required arguments:
            grammar: VRG = the vertex replacement grammar
            u: int = a node covered by the grammar
            v: int = a node covered by the grammar
            mode: str = one of `add` or `del`
            time: int = the timestep when the edge event (u, v) was observed

        Returns:
            The grammar with the mutated rule in it.
    """
    mode = mode.lower()
    assert mode in ('add', 'del')

    parent_idx, parent_metarule, mapping = common_ancestor({u, v}, grammar, t2)

    if mode == 'add':
        parent_metarule[t2].graph.add_edge(mapping[u], mapping[v])
    else:
        parent_metarule[t2].graph.remove_edge(mapping[u], mapping[v])

    if 'label' in parent_metarule[t2].graph.nodes[mapping[u]]:
        uparent_idx, uparent_metarule, ancestor_u = ancestor(u, grammar, t2)
        assert parent_idx != uparent_idx

        if mode == 'add':
            parent_metarule[t2].graph.nodes[mapping[u]]['label'] += 1
            uparent_metarule[t2].graph.nodes[ancestor_u]['b_deg'] += 1
            uparent_metarule[t2].lhs += 1
        else:
            parent_metarule[t2].graph.nodes[mapping[u]]['label'] -= 1
            uparent_metarule[t2].graph.nodes[ancestor_u]['b_deg'] -= 1
            uparent_metarule[t2].lhs -= 1

        propagate_ancestors(grammar.decomposition[uparent_idx][2], grammar.decomposition[uparent_idx][1], grammar, t1, t2, mode, stop_at=parent_idx)

    if 'label' in parent_metarule[t2].graph.nodes[mapping[v]]:
        vparent_idx, vparent_metarule, ancestor_v = ancestor(v, grammar, t2)
        assert parent_idx != vparent_idx

        if mode == 'add':
            parent_metarule[t2].graph.nodes[mapping[v]]['label'] += 1
            vparent_metarule[t2].graph.nodes[ancestor_v]['b_deg'] += 1
            vparent_metarule[t2].lhs += 1
        else:
            parent_metarule[t2].graph.nodes[mapping[v]]['label'] -= 1
            vparent_metarule[t2].graph.nodes[ancestor_v]['b_deg'] -= 1
            vparent_metarule[t2].lhs -= 1

        propagate_ancestors(grammar.decomposition[vparent_idx][2], grammar.decomposition[vparent_idx][1], grammar, t1, t2, mode, stop_at=parent_idx)


def diplomatic(grammar: VRG, u: int, v: int, t1: int, t2: int):  # TODO: update docstring instructions
    """
        Given an edge event (u, v), where u ∈ G but v ∉ G:
            1. Find the lowest rule R that covers u
            2. Update R -> R' by adding v along with an edge (u, v) to the RHS of R
            3. Merge the modified R' back into G
            4. If R was learned at time t: remove R from G; otherwise, preserve R

        Required arguments:
            grammar: VRG = the vertex replacement grammar
            u: int = a node covered by the grammar
            v: int = a node not covered by the grammar
            time: int = the timestep when the edge event (u, v) was observed

        Returns:
            The grammar with the mutated rule in it.
    """
    parent_idx, parent_metarule, ancestor_u = ancestor(u, grammar, t2)
    ancestor_v = chr(ord(max(parent_metarule[t2].graph.nodes())) + 1)

    parent_metarule[t2].graph.add_node(ancestor_v, b_deg=0)
    parent_metarule[t2].graph.add_edge(ancestor_u, ancestor_v)

    if 'label' in parent_metarule[t2].graph.nodes[ancestor_u]:  # if we modified a nonterminal, propagate that change downstream
        parent_metarule[t2].graph.nodes[ancestor_u]['label'] += 1  # one more edge incident on this symbol
        propagate_descendants(ancestor_u, parent_idx, grammar, t1, t2, mode='add')

    grammar.cover[t2][v] = parent_idx  # rule_source now points to the same location in the rule_tree for u and v
    parent_metarule[t2].alias[v] = ancestor_v


def remove_citizen(grammar: VRG, u: int, time: int):  # TODO: update docstring instructions
    _, parent_metarule, ancestor_u = ancestor(u, grammar, time)
    assert parent_metarule[time].graph.nodes[ancestor_u]['b_deg'] == 0
    assert parent_metarule[time].graph.degree(ancestor_u) == 0

    parent_metarule[time].graph.remove_node(ancestor_u)
    del parent_metarule[time].alias[u]
    del grammar.cover[time][u]


def delete_domestic(grammar: VRG, u: int, v: int, t1: int, t2: int):  # TODO: update docstring instructions
    uparent_idx, uparent_metarule, ancestor_u = ancestor(u, grammar, t2)
    vparent_idx, vparent_metarule, ancestor_v = ancestor(v, grammar, t2)

    if uparent_idx == vparent_idx:
        parent_metarule = uparent_metarule
        assert 'label' not in parent_metarule[t2].graph.nodes[ancestor_u]
        assert 'label' not in parent_metarule[t2].graph.nodes[ancestor_v]

        try:
            parent_metarule[t2].graph.remove_edge(ancestor_u, ancestor_v)
        except NetworkXError:
            print(f'the connection {ancestor_u} --- {ancestor_v} cannot be further severed')
    else:
        parent_idx, parent_metarule, mapping = common_ancestor({u, v}, grammar, t2)
        try:
            parent_metarule[t2].graph.remove_edge(mapping[u], mapping[v])
        except NetworkXError:
            print(f'the connection {mapping[u]} --- {mapping[v]} cannot be further severed')

        if 'label' in parent_metarule[t2].graph.nodes[mapping[u]]:
            assert parent_idx != uparent_idx
            parent_metarule[t2].graph.nodes[mapping[u]]['label'] -= 1

            uparent_metarule[t2].lhs -= 1
            uparent_metarule[t2].graph.nodes[ancestor_u]['b_deg'] -= 1
            propagate_ancestors(grammar.decomposition[uparent_idx][2], grammar.decomposition[uparent_idx][1], grammar, t1, t2, mode='del', stop_at=parent_idx)

        if 'label' in parent_metarule[t2].graph.nodes[mapping[v]]:
            parent_metarule[t2].graph.nodes[mapping[v]]['label'] -= 1

            vparent_metarule[t2].lhs -= 1
            vparent_metarule[t2].graph.nodes[ancestor_v]['b_deg'] -= 1
            propagate_ancestors(grammar.decomposition[vparent_idx][2], grammar.decomposition[vparent_idx][1], grammar, t1, t2, mode='del', stop_at=parent_idx)
