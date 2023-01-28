from networkx.exception import NetworkXError

from cnrg.VRG import VRG
from src.bookkeeping import ancestor, common_ancestor, redact, unseal, propagate_ancestors, propagate_descendants


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
    mode = mode.strip().lower()
    assert mode in ('add', 'del')

    uparent_idx, uparent_metarule, ancestor_u = ancestor(u, grammar)
    if ancestor_u not in uparent_metarule[t2].graph:
        if uparent_metarule[t2].graph.order() == 0:
            unseal(grammar,
                   grammar.decomposition[uparent_idx][1],
                   grammar.decomposition[uparent_idx][2],
                   t2)
        uparent_metarule[t2].graph.add_node(ancestor_u, b_deg=0)
        grammar.cover[t2][u] = uparent_idx

    vparent_idx, vparent_metarule, ancestor_v = ancestor(v, grammar)
    if ancestor_v not in vparent_metarule[t2].graph:
        if vparent_metarule[t2].graph.order() == 0:
            unseal(grammar,
                   grammar.decomposition[vparent_idx][1],
                   grammar.decomposition[vparent_idx][2],
                   t2)
        vparent_metarule[t2].graph.add_node(ancestor_v, b_deg=0)
        grammar.cover[t2][v] = vparent_idx

    common_idx, common_metarule, mapping = common_ancestor({u, v}, grammar)
    if common_metarule[t2].graph.order() == 0:
        raise AssertionError(f'rule #{common_metarule.idn} is empty when it should not be')

    if mode == 'add':
        common_metarule[t2].graph.add_edge(mapping[u], mapping[v])
    else:
        common_metarule[t2].graph.remove_edge(mapping[u], mapping[v])

    if 'label' in common_metarule[t2].graph.nodes[mapping[u]]:
        assert common_idx != uparent_idx

        if mode == 'add':
            common_metarule[t2].graph.nodes[mapping[u]]['label'] += 1
            uparent_metarule[t2].graph.nodes[ancestor_u]['b_deg'] += 1
            uparent_metarule[t2].lhs += 1
        else:
            common_metarule[t2].graph.nodes[mapping[u]]['label'] -= 1
            uparent_metarule[t2].graph.nodes[ancestor_u]['b_deg'] -= 1
            uparent_metarule[t2].lhs -= 1

            if uparent_metarule[t2].graph.nodes[ancestor_u]['b_deg'] < 0:
                raise AssertionError

        propagate_ancestors(grammar.decomposition[uparent_idx][2], grammar.decomposition[uparent_idx][1], uparent_metarule[t2].lhs, grammar, t1, t2, mode, stop_at=common_idx)

    if 'label' in common_metarule[t2].graph.nodes[mapping[v]]:
        assert common_idx != vparent_idx

        if mode == 'add':
            common_metarule[t2].graph.nodes[mapping[v]]['label'] += 1
            vparent_metarule[t2].graph.nodes[ancestor_v]['b_deg'] += 1
            vparent_metarule[t2].lhs += 1
        else:
            common_metarule[t2].graph.nodes[mapping[v]]['label'] -= 1
            vparent_metarule[t2].graph.nodes[ancestor_v]['b_deg'] -= 1
            vparent_metarule[t2].lhs -= 1

            if vparent_metarule[t2].graph.nodes[ancestor_v]['b_deg'] < 0:
                raise AssertionError

        propagate_ancestors(grammar.decomposition[vparent_idx][2], grammar.decomposition[vparent_idx][1], vparent_metarule[t2].lhs, grammar, t1, t2, mode, stop_at=common_idx)


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
    parent_idx, parent_metarule, ancestor_u = ancestor(u, grammar)

    if parent_metarule[t2].graph.order() == 0:
        unseal(grammar, grammar.decomposition[parent_idx][1], grammar.decomposition[parent_idx][2], t2)

    if ancestor_u not in parent_metarule[t2].graph:
        parent_metarule[t2].graph.add_node(ancestor_u, b_deg=0)
        grammar.cover[t2][u] = parent_idx

    # ancestor_v = chr(ord(max(parent_metarule[t2].graph.nodes())) + 1)
    # ancestor_v = chr(ord(max(parent_metarule.alias.values())) + 1)
    ancestor_v = parent_metarule.next

    parent_metarule[t2].graph.add_node(ancestor_v, b_deg=0)
    parent_metarule[t2].graph.add_edge(ancestor_u, ancestor_v)
    parent_metarule.alias[v] = ancestor_v
    grammar.cover[t2][v] = parent_idx

    if 'label' in parent_metarule[t2].graph.nodes[ancestor_u]:  # propagate changes downstream
        parent_metarule[t2].graph.nodes[ancestor_u]['label'] += 1  # one more edge incident on this symbol
        propagate_descendants(ancestor_u, parent_idx, grammar, t1, t2, mode='add')


def censor_citizen(grammar: VRG, u: int, time: int):  # TODO: update docstring instructions
    _, parent_metarule, ancestor_u = ancestor(u, grammar)
    assert parent_metarule[time].graph.nodes[ancestor_u]['b_deg'] == 0
    assert parent_metarule[time].graph.degree(ancestor_u) == 0

    parent_metarule[time].graph.remove_node(ancestor_u)
    del grammar.cover[time][u]

    if parent_metarule[time].graph.order() == 0:
        _, pidx, anode = grammar.decomposition[parent_metarule.idn]
        redact(grammar, pidx, anode, time)
