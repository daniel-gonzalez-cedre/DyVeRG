import sys
import numpy as np

sys.path.append('../')

from cnrg.VRG import VRG
from cnrg.Rule import PartRule

from decomposition import ancestor, common_ancestor
from utils import find, timeout


def mutate_rule_domestic(grammar: VRG, u: int, v: int, mode: str, time: int) -> VRG:
    """
        Given an edge event (u, v), where both u and v are already known to the grammar G:
            1. Find the lowest rule R that covers both u and v simultaneously
            2. Update R -> R' by adding or deleting an edge between u and v on the RHS of R
            3. Merge the modified R' back into G
            4. If R was learned at time t: remove R from G; otherwise, preserve R

        Positional arguments:
            grammar: VRG = the vertex replacement grammar
            u: int = a node covered by the grammar
            v: int = a node covered by the grammar
            mode: str = one of `add` or `del`
            time: int = the timestep when the edge event (u, v) was observed

        Returns:
            The grammar with the mutated rule in it.
    """
    parent_rule, which_parent, which_children = common_ancestor({u, v}, grammar)
    which_u = which_children[u]
    which_v = which_children[v]
    ancestor_u = list(parent_rule.graph.nodes())[which_u]
    ancestor_v = list(parent_rule.graph.nodes())[which_v]

    new_rule = parent_rule.copy()
    new_rule.time_changed = time
    if mode == 'add':
        new_rule.graph.add_edge(ancestor_u, ancestor_v)
    elif mode == 'del':
        try:
            new_rule.graph.remove_edge(ancestor_u, ancestor_v)
        except Exception:
            print(f'the connection {ancestor_u} --- {ancestor_v} cannot be further severed')
    else:
        raise AssertionError(f'<<mode>> must be either "add" or "del"; found mode={mode} instead')

    incorporate_rule(grammar, parent_rule, new_rule, which_parent, 1)
    return grammar


def mutate_rule_diplomatic(grammar: VRG, u: int, v: int, time: int) -> VRG:
    """
        Given an edge event (u, v), where u ∈ G but v ∉ G:
            1. Find the lowest rule R that covers u
            2. Update R -> R' by adding v along with an edge (u, v) to the RHS of R
            3. Merge the modified R' back into G
            4. If R was learned at time t: remove R from G; otherwise, preserve R

        Positional arguments:
            grammar: VRG = the vertex replacement grammar
            u: int = a node covered by the grammar
            v: int = a node not covered by the grammar
            time: int = the timestep when the edge event (u, v) was observed

        Returns:
            The grammar with the mutated rule in it.
    """
    parent_rule, which_parent, which_u = ancestor(u, grammar)
    ancestor_u = list(parent_rule.graph.nodes())[which_u]

    new_rule = parent_rule.copy()
    new_rule.time_changed = time
    new_rule.graph.add_node(v, b_deg=0)
    new_rule.graph.add_edge(ancestor_u, v)

    grammar.rule_source[v] = grammar.rule_source[u]  # rule_source now points to the same location in the rule_tree for u and v
    grammar.which_rule_source[v] = list(new_rule.graph.nodes()).index(v)  # compute the pointer for v in the NEW rule's RHS

    incorporate_rule(grammar, parent_rule, new_rule, which_parent, 2)
    return grammar


def incorporate_rule(grammar: VRG, parent_rule: PartRule, new_rule: PartRule, which_parent: int, penalty: int):
    assert penalty in [1, 2]

    # check to see if this rule already exists in the grammar
    parent_idx = find(parent_rule, grammar.rule_list)[0]

    if new_rule.lhs in grammar.rule_dict:
        for rule in grammar.rule_dict[new_rule.lhs]:
            message, is_isomorphic = timeout(lambda x, y: x == y, [new_rule, rule], patience=120)

            # if new_rule is isomorphic to a rule that already exists, merge them
            if not message and is_isomorphic:
                rule.frequency += 1
                grammar.rule_tree[which_parent][0] = rule
                grammar.temporal_matrix[parent_idx, find(rule, grammar.rule_list)[0]] += 1
                return

            # if the isomorphism check times out, mark this rule and find the edit distance
            if message:
                edit_dist = grammar.minimum_edit_dist(new_rule)
                new_rule.edit_dist = edit_dist - penalty
                new_rule.timed_out = True
                print(message)

    # this rule does not already exist in the grammar
    new_rule.edit_dist += penalty
    grammar.replace_rule(parent_rule, new_rule)
    return


def deprecated():
    raise NotImplementedError

    # if mode == 'hash':  # need to evaluate whether or not to deprecate this mode
    #     for parent_idx, other_rule in enumerate(grammar.rule_list):
    #         if parent_rule.hash_equals(other_rule):
    #             break

    #     for new_idx, other_rule in enumerate(grammar.rule_list):
    #         if new_rule.hash_equals(other_rule):
    #             grammar.rule_list[new_idx].frequency += 1
    #             grammar.rule_tree[which_parent][0] = grammar.rule_list[new_idx]
    #             grammar.temporal_matrix[parent_idx, new_idx] += 1  # pylint: disable=undefined-loop-variable
    #             return

    # if decouple:  # if new_rule modifies a rule learned at previous timestep
    #     new_idx = len(grammar.rule_list)  # the index corresponding to the new rule
    #     grammar.temporal_matrix = np.append(grammar.temporal_matrix,
    #                                         np.zeros((1, new_idx)),
    #                                         axis=0)
    #     grammar.temporal_matrix = np.append(grammar.temporal_matrix,
    #                                         np.zeros((new_idx + 1, 1)),
    #                                         axis=1)

    #     grammar.rule_list.append(new_rule)
    #     grammar.rule_tree[which_parent][0] = new_rule

    #     if new_rule.lhs in grammar.rule_dict:
    #         grammar.rule_dict[new_rule.lhs].append(new_rule)
    #     else:
    #         grammar.rule_dict[new_rule.lhs] = [new_rule]

    #     grammar.temporal_matrix[parent_idx, new_idx] += 1
    # else:
    #     grammar.replace_rule(parent_rule, new_rule)
