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

    # for x in (u, v):
    #     xparent_idx, xparent_metarule, ancestor_x = ancestor(x, grammar)

    parent_idx, parent_metarule, mapping = common_ancestor({u, v}, grammar)

    if mode == 'add':  # TODO: think about this; i don't think we ever remove nts, so maybe this is fine?
        if mapping[u] not in parent_metarule[t2].graph:
            parent_metarule[t2].graph.add_node(mapping[u], b_deg=0)
            grammar.cover[t2][u] = parent_idx
        if mapping[v] not in parent_metarule[t2].graph:
            parent_metarule[t2].graph.add_node(mapping[v], b_deg=0)
            grammar.cover[t2][v] = parent_idx
        parent_metarule[t2].graph.add_edge(mapping[u], mapping[v])
    else:
        parent_metarule[t2].graph.remove_edge(mapping[u], mapping[v])

    for x in (u, v):
        if 'label' in parent_metarule[t2].graph.nodes[mapping[x]]:
            xparent_idx, xparent_metarule, ancestor_x = ancestor(x, grammar)
            try:
                assert parent_idx != xparent_idx
            except:
                import pdb
                pdb.set_trace()

            if ancestor_x not in xparent_metarule[t2].graph:
                xparent_metarule[t2].graph.add_node(ancestor_x, b_deg=0)
                grammar.cover[t2][x] = xparent_idx

            if mode == 'add':
                parent_metarule[t2].graph.nodes[mapping[x]]['label'] += 1
                xparent_metarule[t2].graph.nodes[ancestor_x]['b_deg'] += 1
                xparent_metarule[t2].lhs += 1
            else:
                parent_metarule[t2].graph.nodes[mapping[x]]['label'] -= 1
                xparent_metarule[t2].graph.nodes[ancestor_x]['b_deg'] -= 1
                xparent_metarule[t2].lhs -= 1

                if xparent_metarule[t2].graph.nodes[ancestor_x]['b_deg'] < 0:
                    import pdb
                    pdb.set_trace()

            propagate_ancestors(grammar.decomposition[xparent_idx][2], grammar.decomposition[xparent_idx][1], xparent_metarule[t2].lhs, grammar, t1, t2, mode, stop_at=parent_idx)


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


def remove_citizen(grammar: VRG, u: int, time: int):  # TODO: update docstring instructions
    _, parent_metarule, ancestor_u = ancestor(u, grammar)
    try:
        assert parent_metarule[time].graph.nodes[ancestor_u]['b_deg'] == 0
        assert parent_metarule[time].graph.degree(ancestor_u) == 0
    except:
        import pdb
        pdb.set_trace()

    parent_metarule[time].graph.remove_node(ancestor_u)
    del grammar.cover[time][u]
    # del parent_metarule[time].alias[u]
