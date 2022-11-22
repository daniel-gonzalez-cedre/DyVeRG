import sys

sys.path.append('..')

import networkx as nx
from joblib import Parallel, delayed

from cnrg.VRG import VRG
from cnrg.Rule import PartRule
from decomposition import ancestor, common_ancestor
from utils import graph_edit_distance


def join_grammars(home_grammar: VRG, away_grammar: VRG, frontier: set[tuple[int, int]]) -> VRG:
    """
        Joins two grammars in such a way that one grammar is `conditioned` on the other.

        Positional arguments:
            home_grammar: VRG = the grammar treated as a prior
            away_grammar: VRG = the grammar treated as an update
            frontier: {(int, int) for ...} = a set of edges forming a cut on the input graph such that
                                             the two halves of the graph induced partition the nodes
                                             based on which of the two grammars they are covered by
            root: int = the LHS of the root of the decomposition

        Returns:
            The joined grammar.
    """
    home_grammar, away_grammar = (home_grammar.copy(), away_grammar.copy())
    if len(frontier) == 0:
        joined_grammar = conjoin_grammars(home_grammar, away_grammar)
    else:
        prior_grammar, mod_grammar = incise_grammars(home_grammar, away_grammar, frontier)
        joined_grammar = suture_grammars(prior_grammar, mod_grammar)

    return joined_grammar


# when the frontier is empty
def conjoin_grammars(home_grammar: VRG, away_grammar: VRG, parallel: bool = True) -> VRG:
    assert min(away_grammar.rule_dict.keys()) == 0
    assert min(home_grammar.rule_dict.keys()) <= 0

    S = min(home_grammar.rule_dict.keys())
    existed = (S == 0)
    old = chr(0)

    # in the future: to distinguish the different connected components,
    # label the nodes on this splitting rule and modify the generator
    if not existed:  # create a splitting rule
        new = chr(1)
        rhs = nx.Graph()
        rhs.add_node(old,
                     b_deg=0,
                     label=min(home_grammar.rule_dict.keys()))
        rhs.add_node(new,
                     b_deg=0,
                     label=min(away_grammar.rule_dict.keys()))
        splitting_rule = PartRule(S - 1, rhs)
        root_idx = 0
    else:  # splitting rule already exists, so just modify it
        splitting_rule, = home_grammar.rule_dict[S]
        splitting_nodes = nx.convert_node_labels_to_integers(splitting_rule.graph).nodes()
        new = chr(max(v for v in splitting_nodes) + 1)
        splitting_rule.graph.add_node(new,
                                      b_deg=0,
                                      label=min(away_grammar.rule_dict.keys()))

        for idx, (rule, _, _) in enumerate(home_grammar.rule_tree):
            if splitting_rule == rule:
                root_idx = idx
                break
        else:
            raise AssertionError

    offset = len(home_grammar.rule_tree) + (1 if not existed else 0)

    if not existed:
        # shift the indices of the home decomposition if necessary
        for idx, _ in enumerate(home_grammar.rule_tree):
            if home_grammar.rule_tree[idx][1] is not None:
                home_grammar.rule_tree[idx][1] += 1
            else:
                home_grammar.rule_tree[idx][1] = root_idx
                home_grammar.rule_tree[idx][2] = list(splitting_rule.graph.nodes()).index(old)

    # shift the indices of the away decomposition
    for idx, _ in enumerate(away_grammar.rule_tree):
        if away_grammar.rule_tree[idx][1] is not None:
            away_grammar.rule_tree[idx][1] += offset
        else:
            away_grammar.rule_tree[idx][1] = root_idx
            away_grammar.rule_tree[idx][2] = list(splitting_rule.graph.nodes()).index(new)

    if not existed:
        # shift the indices of the home rule_source if necessary
        for idx in home_grammar.rule_source:
            home_grammar.rule_source[idx] += 1

    # shift the indices of the away rule_source
    for idx in away_grammar.rule_source:
        away_grammar.rule_source[idx] += offset

    if not existed:
        # APPEND the rule tree, so that it is a branch under the home decomposition
        # if we were to PREPEND instead, the common_ancestor(...) would no longer work
        home_grammar.rule_tree = [[splitting_rule, None, None]] + home_grammar.rule_tree + away_grammar.rule_tree

    # merge the grammars' rules by combining isomorphic rules together
    for away_rule in away_grammar.rule_list:
        try:  # try to find a rule isomorphic to away_rule
            found_idx = home_grammar.rule_list.index(away_rule)
            home_grammar.rule_list[found_idx].frequency += away_rule.frequency
        except ValueError:  # no such rule found; away_rule must be new
            edit_dist = home_grammar.minimum_edit_dist(away_rule)
            away_rule.edit_dist = edit_dist

            home_grammar.num_rules += 1
            home_grammar.rule_list += [away_rule]

            if away_rule.lhs in home_grammar.rule_dict:
                home_grammar.rule_dict[away_rule.lhs] += [away_rule]
            else:
                home_grammar.rule_dict[away_rule.lhs] = [away_rule]

    # merge the bookkeeping dicts; the node sets should be disjoint, so this is fine
    home_grammar.rule_source |= away_grammar.rule_source
    home_grammar.which_rule_source |= away_grammar.which_rule_source

    # rule_source & which_rule_source do not need to be modified because splitting_rule is not a leaf rule
    if not existed:  # incorporate the new splitting rule
        home_grammar.num_rules += 1
        home_grammar.rule_list += [splitting_rule]
        home_grammar.rule_dict[S] = [splitting_rule]

    return home_grammar


# when the frontier is nonempty
def incise_grammars(home_grammar: VRG, away_grammar: VRG, frontier: set[tuple[int, int]]) -> tuple[VRG, VRG]:
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
    except Exception as e:
        for u, v in frontier:
            if u not in home_grammar.rule_source:
                print(u, u in home_grammar.rule_source, u in away_grammar.rule_source, v, v in home_grammar.rule_source, v in away_grammar.rule_source)
        raise AssertionError(home_grammar.rule_source) from e

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
    # home_grammar.rule_dict = {}
    # for rule in home_grammar.rule_list:
    #     if rule.lhs in home_grammar.rule_dict.keys():
    #         home_grammar.rule_dict[rule.lhs] += [rule]
    #     else:
    #         home_grammar.rule_dict[rule.lhs] = [rule]

    # recompute the rule_dict for away_grammar
    # away_grammar.rule_dict = {}
    # for rule in away_grammar.rule_list:
    #     if rule.lhs in away_grammar.rule_dict.keys():
    #         away_grammar.rule_dict[rule.lhs] += [rule]
    #     else:
    #         away_grammar.rule_dict[rule.lhs] = [rule]

    # refresh the object references
    # this must be done since we just modified some of the rules in preparation for surgery
    # home_grammar.rule_list = list({rule for rule, _, _ in home_grammar.rule_tree})
    # away_grammar.rule_list = list({rule for rule, _, _ in away_grammar.rule_tree})

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
def suture_grammars(home_grammar: VRG, away_grammar: VRG, parallel: bool = True) -> VRG:
    # merge in new rules that are duplicates of old rules
    replacement_pairs = []
    for away_rule in away_grammar.rule_list:
        try:  # if we find an isomorphic rule, combine them and increment the frequency
            found_idx = home_grammar.rule_list.index(away_rule)
            home_rule = home_grammar.rule_list[found_idx]
            home_rule.frequency += 1
            replacement_pairs += [(away_rule, home_rule)]
        except ValueError:  # if there are no isomorphic rules, add this as a new rule
            candidates = [] if away_rule.lhs not in home_grammar.rule_dict else [rule
                                                                                 for rule in home_grammar.rule_dict[away_rule.lhs]
                                                                                 if rule is not away_rule]

            if candidates:
                if parallel:
                    edit_dists = Parallel(n_jobs=4)(
                        delayed(graph_edit_distance)(home_rule.graph, away_rule.graph)
                        for home_rule in candidates
                    )
                else:
                    edit_dists = [graph_edit_distance(home_rule.graph, away_rule.graph)
                                  for home_rule in candidates]
                away_rule.edit_dist = int(min(edit_dists))
            else:
                away_rule.edit_dist = 0

            home_grammar.num_rules += 1
            home_grammar.rule_list += [away_rule]

            if away_rule.lhs in home_grammar.rule_dict:
                home_grammar.rule_dict[away_rule.lhs] += [away_rule]
            else:
                home_grammar.rule_dict[away_rule.lhs] = [away_rule]

    # replace the object pointers for the isomorphic rules
    for away_rule, home_rule in replacement_pairs:
        away_grammar.replace_rule(away_rule, home_rule)

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
    # for tree_idx, (rule, _, _) in enumerate(home_grammar.rule_tree):
    #     try:
    #         list_idx = home_grammar.rule_list.index(rule)
    #         home_grammar.rule_tree[tree_idx][0] = home_grammar.rule_list[list_idx]
    #     except Exception as e:
    #         print(tree_idx, rule)
    #         raise Exception from e

    # merge the bookkeeping dicts
    # the node sets should be disjoint, so this is fine
    home_grammar.rule_source |= away_grammar.rule_source
    home_grammar.which_rule_source |= away_grammar.which_rule_source

    return home_grammar
