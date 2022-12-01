import sys

sys.path.append('../')

from cnrg.VRG import VRG

from decomposition import assimilate_rule, ancestor, common_ancestor, propagate_descendants


# TODO: update docstring instructions
def mutate_rule_domestic(grammar: VRG, u: int, v: int, time: int, mode: str = 'add'):
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
    parent_rule, parent_idx, ancestor_nodes = common_ancestor({u, v}, grammar)
    ancestor_u = ancestor_nodes[u]
    ancestor_v = ancestor_nodes[v]

    if parent_rule.frequency == 1:
        mutated_rule = parent_rule

        list_idx = grammar.find_rule(parent_rule, where='rule_list')
        del grammar.rule_list[list_idx]

        lhs, dict_idx = grammar.find_rule(parent_rule, where='rule_dict')
        del grammar.rule_dict[lhs][dict_idx]

        if len(grammar.rule_dict[lhs]) == 0:
            del grammar.rule_dict[lhs]
    else:
        mutated_rule = parent_rule.copy()
        mutated_rule.edit_dist = 0
        mutated_rule.frequency = 1
        parent_rule.frequency -= 1
        grammar.rule_tree[parent_idx][0] = mutated_rule

    mutated_rule.edit_dist += 1  # cost of adding/removing an edge
    mutated_rule.time_changed = time

    if mode == 'add':
        mutated_rule.graph.add_edge(ancestor_u, ancestor_v)

        for ancestor_x in (ancestor_u, ancestor_v):
            if 'label' in mutated_rule.graph.nodes[ancestor_x]:  # if we modified a nonterminal, propagate that change downstream
                mutated_rule.edit_dist += 1  # cost of relabeling a node
                mutated_rule.graph.nodes[ancestor_x]['label'] += 1  # one more edge incident on this symbol
                propagate_descendants(ancestor_x, parent_idx, grammar, time)
    elif mode == 'del':
        try:
            mutated_rule.graph.remove_edge(ancestor_u, ancestor_v)
        except Exception:
            print(f'the connection {ancestor_u} --- {ancestor_v} cannot be further severed')
    else:
        raise AssertionError(f'<<mode>> must be either "add" or "del"; found mode={mode} instead')

    assimilate_rule(mutated_rule, grammar)


def mutate_rule_diplomatic(grammar: VRG, u: int, v: int, time: int):
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
    parent_rule, parent_idx, ancestor_u = ancestor(u, grammar)

    if parent_rule.frequency == 1:
        mutated_rule = parent_rule

        list_idx = grammar.find_rule(parent_rule, where='rule_list')
        del grammar.rule_list[list_idx]

        lhs, dict_idx = grammar.find_rule(parent_rule, where='rule_dict')
        del grammar.rule_dict[lhs][dict_idx]

        if len(grammar.rule_dict[lhs]) == 0:
            del grammar.rule_dict[lhs]
    else:
        mutated_rule = parent_rule.copy()
        mutated_rule.frequency = 1
        parent_rule.frequency -= 1
        grammar.rule_tree[parent_idx][0] = mutated_rule

    ancestor_v = chr(ord(max(mutated_rule.graph.nodes())) + 1)
    # mutated_rule.mapping[v] = ancestor_v

    mutated_rule.edit_dist += 2  # cost of adding a node and an edge
    mutated_rule.time_changed = time
    mutated_rule.graph.add_node(ancestor_v, b_deg=0, look='at me')
    mutated_rule.graph.add_edge(ancestor_u, ancestor_v)

    if 'label' in mutated_rule.graph.nodes[ancestor_u]:  # if we modified a nonterminal, propagate that change downstream
        mutated_rule.edit_dist += 1  # cost of relabeling a node
        mutated_rule.graph.nodes[ancestor_u]['label'] += 1  # one more edge incident on this symbol
        propagate_descendants(ancestor_u, parent_idx, grammar, time)

    grammar.covering_idx[v] = parent_idx  # rule_source now points to the same location in the rule_tree for u and v

    assimilated_rule, f = assimilate_rule(mutated_rule, grammar)
    assimilated_rule.mapping[v] = f[ancestor_v]
