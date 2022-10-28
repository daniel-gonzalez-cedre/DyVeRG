import sys
from typing import Set, Tuple

sys.path.append('..')

import networkx as nx

from cnrg.VRG import VRG
from cnrg.Rule import PartRule
from bookkeeping import ancestor, common_ancestor


def graft_grammars(home_grammar: VRG, away_grammar: VRG, frontier: Set[Tuple[int, int]]):
    home_grammar, away_grammar = (home_grammar.copy(), away_grammar.copy())

    if len(frontier) == 0:
        stitched_grammar = conjoin_grammars(home_grammar, away_grammar)
    else:
        prior_grammar, mod_grammar = incise_grammars(home_grammar, away_grammar, frontier)
        stitched_grammar = suture_grammars(prior_grammar, mod_grammar)

    return stitched_grammar


# when the frontier is empty
def conjoin_grammars(home_grammar: VRG, away_grammar: VRG):
    S = min(home_grammar.rule_dict.keys() | away_grammar.rule_dict.keys()) - 1
    rhs = nx.Graph()
    rhs.add_node(chr(0), b_deg=0, label=min(home_grammar.rule_dict.keys()))
    rhs.add_node(chr(1), b_deg=0, label=min(away_grammar.rule_dict.keys()))

    splitting_rule = PartRule(S, rhs)

    offset = len(home_grammar.rule_tree) + 1

    # shift the indices of the decompositions
    for idx, _ in enumerate(home_grammar.rule_tree):
        if home_grammar.rule_tree[idx][1] is not None:
            home_grammar.rule_tree[idx][1] += 1
        else:
            home_grammar.rule_tree[idx][1] = 0
            home_grammar.rule_tree[idx][2] = list(splitting_rule.graph.nodes()).index(chr(0))

    for idx, _ in enumerate(away_grammar.rule_tree):
        if away_grammar.rule_tree[idx][1] is not None:
            away_grammar.rule_tree[idx][1] += offset
        else:
            away_grammar.rule_tree[idx][1] = 0
            away_grammar.rule_tree[idx][2] = list(splitting_rule.graph.nodes()).index(chr(1))

    # shift the indices of the rule_source maps
    for idx in home_grammar.rule_source:
        home_grammar.rule_source[idx] += 1

    for idx in away_grammar.rule_source:
        away_grammar.rule_source[idx] += offset

    # APPEND the rule tree, so that it is a branch under the home decomposition
    # if we were to PREPEND instead, the common_ancestor(...) would no longer work
    home_grammar.rule_tree = [[splitting_rule, None, None]] + home_grammar.rule_tree + away_grammar.rule_tree

    # merge in new rules that are duplicates of old rules
    for away_rule in away_grammar.rule_list:
        try:
            found_idx = home_grammar.rule_list.index(away_rule)
            home_grammar.rule_list[found_idx].frequency += 1
        except ValueError:
            home_grammar.num_rules += 1
            home_grammar.rule_list += [away_rule]

            if away_rule.lhs in home_grammar.rule_dict:
                home_grammar.rule_dict[away_rule.lhs] += [away_rule]
            else:
                home_grammar.rule_dict[away_rule.lhs] = [away_rule]

    # merge the bookkeeping dicts
    # the node sets should be disjoint, so this is fine
    home_grammar.rule_source |= away_grammar.rule_source
    home_grammar.which_rule_source |= away_grammar.which_rule_source

    # incorporate the new splitting rule
    home_grammar.rule_list += [splitting_rule]
    home_grammar.rule_dict[S] = [splitting_rule]
    # rule_source does not need to be modified because splitting_rule is not a leaf rule
    # which_rule_source does not need to be modified for the same reason

    return home_grammar


# when the frontier is nonempty
def incise_grammars(home_grammar: VRG, away_grammar: VRG, frontier: Set[Tuple[int, int]]):
    # u is old => covered by the home_grammar
    # v is new => covered by the away_grammar
    frontier = {(u if u in home_grammar.rule_source else v, v if u in home_grammar.rule_source else u)
                for u, v in frontier}

    # adding new boundary edges to represent the frontier
    for _, v in frontier:
        covering_rule_v, _, which_v = ancestor(v, away_grammar)
        _, covered_d = list(covering_rule_v.graph.nodes(data=True))[which_v]
        covered_d['b_deg'] += 1

        if covering_rule_v.lhs != 0:
            covering_rule_v.lhs += 1

    # find the root rule for the new branch (where away_grammar will be grafted in under)
    try:
        covering_rule, which_cover, which_children = common_ancestor({u for u, _ in frontier}, home_grammar)
    except Exception:
        for u, v in frontier:
            if u not in home_grammar.rule_source:
                print(u, u in home_grammar.rule_source, u in away_grammar.rule_source, v, v in home_grammar.rule_source, v in away_grammar.rule_source)
        # print(home_grammar.rule_source)
        exit()
    nts = chr(ord(max(covering_rule.graph.nodes())) + 1)

    # add the nonterminal symbol and connect it to the rest of this root
    covering_rule.graph.add_node(nts, label=len(frontier), b_deg=0)
    for u, _ in frontier:
        which_u = which_children[u]
        covering_u, _ = list(covering_rule.graph.nodes(data=True))[which_u]
        if covering_rule.graph.has_edge(covering_u, nts):
            covering_rule.graph.edges[covering_u, nts]['weight'] += 1
        else:
            covering_rule.graph.add_edge(covering_u, nts, weight=1)

    # recompute home_grammar.rule_dict
    home_grammar.rule_dict = {}
    for rule in home_grammar.rule_list:
        if rule.lhs in home_grammar.rule_dict.keys():
            home_grammar.rule_dict[rule.lhs] += [rule]
        else:
            home_grammar.rule_dict[rule.lhs] = [rule]

    # recompute the rule_dict for away_grammar
    away_grammar.rule_dict = {}
    for rule in away_grammar.rule_list:
        if rule.lhs in away_grammar.rule_dict.keys():
            away_grammar.rule_dict[rule.lhs] += [rule]
        else:
            away_grammar.rule_dict[rule.lhs] = [rule]

    # refresh the object references
    # this must be done since we just modified some of the rules in preparation for surgery
    home_grammar.rule_list = list({rule for rule, _, _ in home_grammar.rule_tree})
    away_grammar.rule_list = list({rule for rule, _, _ in away_grammar.rule_tree})

    # find the root of the decomposition in the away_grammar
    root_lhs = min(rule.lhs for (rule, _, _) in away_grammar.rule_tree)
    for idx, (rule, _, _) in enumerate(away_grammar.rule_tree):
        if rule.lhs == root_lhs:
            root_rule = rule
            root_idx = idx

    # shift the indices of the away decomposition
    offset = len(home_grammar.rule_tree)
    for idx, _ in enumerate(away_grammar.rule_tree):
        if away_grammar.rule_tree[idx][1] is not None:
            away_grammar.rule_tree[idx][1] += offset

    # shift the indices of the rule_source map
    for idx in away_grammar.rule_source:
        away_grammar.rule_source[idx] += offset

    # change the size of the root rule for the new branch
    root_rule.lhs = len(frontier)

    # recompute the parent for the root rule in the new decomposition
    assert away_grammar.rule_tree[root_idx][1] is None
    away_grammar.rule_tree[root_idx][1] = which_cover
    away_grammar.rule_tree[root_idx][2] = list(covering_rule.graph.nodes()).index(nts)

    return home_grammar, away_grammar


# when the frontier is nonempty
# def suture_grammars(home_grammar: VRG, away_grammar: VRG, frontier: Set[Tuple[int, int]], mode: str = 'hash'):
def suture_grammars(home_grammar: VRG, away_grammar: VRG):
    # assert mode in ['iso', 'hash']

    # merge in new rules that are duplicates of old rules
    for away_rule in away_grammar.rule_list:
        try:
            found_idx = home_grammar.rule_list.index(away_rule)
            home_grammar.rule_list[found_idx].frequency += 1
        except ValueError:
            home_grammar.num_rules += 1
            home_grammar.rule_list += [away_rule]

            if away_rule.lhs in home_grammar.rule_dict:
                home_grammar.rule_dict[away_rule.lhs] += [away_rule]
            else:
                home_grammar.rule_dict[away_rule.lhs] = [away_rule]

    # # modify the decomposition
    # # shift the indices of the away decomposition
    # offset = len(home_grammar.rule_tree)
    # for idx in range(len(away_grammar.rule_tree)):
    #     away_grammar.rule_tree[idx][1] += offset

    # APPEND the rule tree, so that it is a branch under the home decomposition
    # if we were to PREPEND instead, the common_ancestor(...) would no longer work
    home_grammar.rule_tree = home_grammar.rule_tree + away_grammar.rule_tree
    # home_grammar.rule_tree = away_grammar.rule_tree + home_grammar.rule_tree

    # recompute grammar references in rule_tree
    for tree_idx, (rule, _, _) in enumerate(home_grammar.rule_tree):
        try:
            list_idx = home_grammar.rule_list.index(rule)
            home_grammar.rule_tree[tree_idx][0] = home_grammar.rule_list[list_idx]
        except Exception as e:
            print(tree_idx, rule)
            raise Exception from e

    # merge the bookkeeping dicts
    # the node sets should be disjoint, so this is fine
    home_grammar.rule_source |= away_grammar.rule_source
    home_grammar.which_rule_source |= away_grammar.which_rule_source

    # home_grammar.recalculate_cost()

    return home_grammar
