import sys
import random

sys.path.append('../')

from cnrg.VRG import VRG
from cnrg.Rule import PartRule

from decomposition import ancestor, common_ancestor
from utils import is_rule_isomorphic


def assimilate(new_rule: PartRule, grammar: VRG):
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
                    break
            else:
                found = True
        else:
            if not found:
                grammar.rule_dict[new_rule.lhs] += [new_rule]
                grammar.rule_list += [new_rule]
    else:
        grammar.rule_dict[new_rule.lhs] = [new_rule]
        grammar.rule_list += [new_rule]


def modify_descendants(nts: str, rule_idx: int, grammar: VRG, time: int):
    for child_idx, child_rule in grammar.get_children_of(nts, rule_idx):
        if child_rule.frequency == 1:
            modified_rule = child_rule

            list_idx = grammar.find_rule(child_rule, where='rule_list')
            del grammar.rule_list[list_idx]

            lhs, dict_idx = grammar.find_rule(child_rule, where='rule_dict')
            del grammar.rule_dict[lhs][dict_idx]

            if len(grammar.rule_dict[lhs]) == 0:
                del grammar.rule_dict[lhs]
        else:
            modified_rule = child_rule.copy()
            modified_rule.edit_dist = 0
            modified_rule.frequency = 1
            child_rule.frequency -= 1
            grammar.rule_tree[child_idx][0] = modified_rule

        (v, d), = random.sample(modified_rule.graph.nodes(data=True), 1)
        d['b_deg'] += 1

        if 'label' in d:
            d['label'] += 1
            modified_rule.edit_dist += 1  # cost of relabeling a node

            modify_descendants(v, child_idx, grammar, time)

        modified_rule.lhs += 1
        modified_rule.edit_dist += 2  # cost of relabeling a node and changing the RHS
        modified_rule.time_changed = time

        assimilate(modified_rule, grammar)


# TODO: update docstring instructions
def mutate_rule():
    raise NotImplementedError


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
                modify_descendants(ancestor_x, parent_idx, grammar, time)
    elif mode == 'del':
        try:
            mutated_rule.graph.remove_edge(ancestor_u, ancestor_v)
        except Exception:
            print(f'the connection {ancestor_u} --- {ancestor_v} cannot be further severed')
    else:
        raise AssertionError(f'<<mode>> must be either "add" or "del"; found mode={mode} instead')

    assimilate(mutated_rule, grammar)


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

    mutated_rule.edit_dist += 2  # cost of adding a node and an edge
    mutated_rule.time_changed = time
    mutated_rule.graph.add_node(v, b_deg=0, look='at me')
    mutated_rule.graph.add_edge(ancestor_u, v)

    if 'label' in mutated_rule.graph.nodes[ancestor_u]:  # if we modified a nonterminal, propagate that change downstream
        mutated_rule.edit_dist += 1  # cost of relabeling a node
        mutated_rule.graph.nodes[ancestor_u]['label'] += 1  # one more edge incident on this symbol
        modify_descendants(ancestor_u, parent_idx, grammar, time)

    grammar.covering_idx[v] = grammar.covering_idx[u]  # rule_source now points to the same location in the rule_tree for u and v

    assimilate(mutated_rule, grammar)
