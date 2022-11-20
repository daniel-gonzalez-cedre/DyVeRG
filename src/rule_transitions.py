import sys
import copy
import numpy as np
# import networkx.algorithms.isomorphism as iso

sys.path.append('../')

from cnrg.VRG import VRG

from bookkeeping import ancestor, common_ancestor


# at time t: node u exists, node v also exists
def mutate_rule_domestic(grammar: VRG, u: int, v: int, mode: str, time: int):
    """
        Given an edge event (u, v), where both u and v are already known to the grammar G:
            1. Find the lowest rule R that covers both u and v simultaneously
            2. Update R -> R' by adding or deleting an edge between u and v on the RHS of R
            3. Merge the modified R' back into G
            4. If R was learned at time t: remove R from G; otherwise, preserve R

        Positional arguments:
            grammar [VRG]: the vertex replacement grammar
            u [int]: a node covered by the grammar
            v [int]: a node covered by the grammar
            mode [str]: one of `add` or `del`
            time [int]: the timestep when the edge event (u, v) was observed
    """
    parent_rule, which_parent, which_children = common_ancestor({u, v}, grammar)
    which_u = which_children[u]
    which_v = which_children[v]
    ancestor_u = list(parent_rule.graph.nodes())[which_u]
    ancestor_v = list(parent_rule.graph.nodes())[which_v]

    # if the rule to be modified was learned at a prior timestep, preserve it
    # otherwise, directly modify that rule and overwrite it
    if parent_rule.time < time:
        new_rule = copy.deepcopy(parent_rule)
        new_rule.time = time
    else:
        new_rule = parent_rule

    if mode == 'add':
        new_rule.graph.add_edge(ancestor_u, ancestor_v)
    elif mode == 'del':
        try:
            new_rule.graph.edges[ancestor_u, ancestor_v]['weight'] -= 1
            if new_rule.graph.edges[ancestor_u, ancestor_v]['weight'] == 0:
                new_rule.graph.remove_edge(ancestor_u, ancestor_v)
        except Exception:
            print(f'the connection {ancestor_u} --- {ancestor_v} cannot be further severed')
    else:
        raise AssertionError('<<mode>> must be either "add" or "del" ' +
                             f'found mode={mode} instead')

    if parent_rule.time < time:
        _, parent_idx, new_idx = incorporate_new_rule(grammar, parent_rule, new_rule, which_parent)
        # grammar.temporal_matrix[parent_idx, parent_idx] = max(0, grammar.temporal_matrix[parent_idx, parent_idx] - 1)
        grammar.temporal_matrix[parent_idx, new_idx] += 1

    return grammar


# at time t: node u exists, node v does not exist
def mutate_rule_diplomatic(grammar: VRG, u: int, v: int, time: int):
    """
        Given an edge event (u, v), where u ∈ G but v ∉ G:
            1. Find the lowest rule R that covers u
            2. Update R -> R' by adding v along with an edge (u, v) to the RHS of R
            3. Merge the modified R' back into G
            4. If R was learned at time t: remove R from G; otherwise, preserve R

        Positional arguments:
            grammar [VRG]: the vertex replacement grammar
            u [int]: a node covered by the grammar
            v [int]: a node not covered by the grammar
            time [int]: the timestep when the edge event (u, v) was observed
    """
    parent_rule, which_parent, which_u = ancestor(u, grammar)
    ancestor_u = list(parent_rule.graph.nodes())[which_u]

    # if the rule to be modified was learned at a prior timestep, preserve it
    # otherwise, directly modify that rule and overwrite it
    if parent_rule.time < time:
        new_rule = copy.deepcopy(parent_rule)
        new_rule.time = time
    else:
        new_rule = parent_rule

    new_rule.graph.add_node(v, b_deg=0)

    # there is no need to check for modes since this can only possibly be an edge addition
    new_rule.graph.add_edge(ancestor_u, v)

    # if mode == 'add':
    #     new_rule.graph.add_edge(ancestor_u, v)
    # elif mode == 'del':
    #     # is this even possible? i don't think so
    #     raise AssertionError('!!! PANIC !!! encountered <<del>> when adding a new node !!! PANIC !!!')
    #     # new_rule.graph.add_edge(ancestor_u, v)
    # else:
    #     raise AssertionError('<<mode>> must be either "add" or "del" ' +
    #                          f'found mode={mode} instead')

    grammar.rule_source[v] = grammar.rule_source[u]  # rule_source now points to the same location in the rule_tree for u and v
    # grammar.which_rule_source[u] = list(rule.graph.nodes()).index(u)  # recompute the pointer for u in the rule's RHS (THIS LINE MIGHT NOT MAKE SENSE!!)
    grammar.which_rule_source[v] = list(new_rule.graph.nodes()).index(v)  # compute the pointer for v in the NEW rule's RHS

    # if parent_rule.time < time:
    _, parent_idx, new_idx = incorporate_new_rule(grammar, parent_rule, new_rule, which_parent)

    # parent_idx = grammar.rule_list.index(parent_rule)
    # new_idx = grammar.rule_list.index(new_rule)

    # grammar.temporal_matrix[parent_idx, parent_idx] = max(0, grammar.temporal_matrix[parent_idx, parent_idx] - 1)
    grammar.temporal_matrix[parent_idx, new_idx] += 1

    return grammar


def incorporate_new_rule(grammar, parent_rule, new_rule, which_parent, mode='hash'):
    assert mode in ['iso', 'hash']

    if mode == 'iso':  # # this one is working; the other one isn't (out of bounds index `which_u`, don't know why yet)
        parent_idx = grammar.rule_list.index(parent_rule)

        if new_rule in grammar.rule_dict[new_rule.lhs]:
            new_idx = grammar.rule_list.index(new_rule)
            grammar.rule_list[new_idx].frequency += 1
            grammar.rule_tree[which_parent][0] = grammar.rule_list[new_idx]
            return grammar.rule_list[new_idx], parent_idx, new_idx
    elif mode == 'hash':
        for parent_idx, other_rule in enumerate(grammar.rule_list):
            if parent_rule.hash_equals(other_rule):
                break

        for new_idx, other_rule in enumerate(grammar.rule_list):
            if new_rule.hash_equals(other_rule):
                grammar.rule_list[new_idx].frequency += 1
                grammar.rule_tree[which_parent][0] = grammar.rule_list[new_idx]

                return grammar.rule_list[new_idx], parent_idx, new_idx
    else:
        raise NotImplementedError(f'mode {mode} not recognized')

    # if the rule is new
    new_rule.edit_cost += 1
    new_idx = len(grammar.rule_list)  # the index corresponding to the new rule
    grammar.temporal_matrix = np.append(grammar.temporal_matrix,
                                        np.zeros((1, new_idx)),
                                        axis=0)
    grammar.temporal_matrix = np.append(grammar.temporal_matrix,
                                        np.zeros((new_idx + 1, 1)),
                                        axis=1)

    grammar.rule_list.append(new_rule)
    grammar.rule_tree[which_parent][0] = new_rule
    # grammar.rule_tree[which_parent][2] = HERE

    if new_rule.lhs in grammar.rule_dict:
        grammar.rule_dict[new_rule.lhs].append(new_rule)
    else:
        grammar.rule_dict[new_rule.lhs] = [new_rule]

    return grammar.rule_list[new_idx], parent_idx, new_idx
