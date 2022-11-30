import sys

sys.path.append('..')

import networkx as nx
from joblib import Parallel, delayed

from cnrg.VRG import VRG
from decomposition import create_splitting_rule, ancestor, common_ancestor
from utils import graph_edit_distance, is_rule_isomorphic


# merge in new rules that are duplicates of old rules
def merge_rules(home_grammar: VRG, away_grammar: VRG, parallel: bool = True):
    away_tree = away_grammar.rule_tree.copy()
    for away_rule, _, _ in away_tree:
        if not home_grammar.find_rule(away_rule, where='rule_list'):
            if away_rule.lhs in home_grammar.rule_dict:
                for superrule in home_grammar.rule_dict[away_rule.lhs]:
                    if f := is_rule_isomorphic(away_rule, superrule):
                        superrule.frequency += away_rule.frequency
                        superrule.subtree |= away_rule.subtree

                        for u in away_rule.mapping:
                            superrule.mapping[u] = f[away_rule.mapping[u]]

                        away_grammar.replace_rule(away_rule, superrule, f)

                        break
                else:
                    away_rule.edit_dist = home_grammar.minimum_edit_dist(away_rule, parallel=parallel)
                    home_grammar.rule_list += [away_rule]

                    home_grammar.rule_dict[away_rule.lhs] += [away_rule]
            else:
                home_grammar.rule_dict[away_rule.lhs] = [away_rule]


def join_grammars(home_grammar: VRG, away_grammar: VRG, frontier: set[tuple[int, int]]) -> VRG:
    """
        Joins two grammars in such a way that one grammar is `conditioned` on the other.

        Required arguments:
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

    prior_grammar, mod_grammar, cover_idx, away_node = incise_grammars(home_grammar, away_grammar, frontier)
    joined_grammar = suture_grammars(prior_grammar, mod_grammar, cover_idx, away_node)

    return joined_grammar


# when the frontier is empty
def conjoin_grammars(home_grammar: VRG, away_grammar: VRG, parallel: bool = True) -> VRG:
    assert min(lhs for lhs in away_grammar.rule_dict) == 0
    assert min(lhs for lhs in home_grammar.rule_dict) <= 0

    time = max(rule.time for rule in home_grammar.rule_list + away_grammar.rule_list)
    S = min(lhs for lhs in home_grammar.rule_dict)

    if S == 0:  # create & incorporate a splitting rule
        splitting_rule = create_splitting_rule([home_grammar, away_grammar], time)
        splitting_idx = len(home_grammar.rule_tree)
        home_node = '0'
        away_node = '1'

        # make the root of the old decomposition point to the new root
        for idx, (_, parent_idx, ancestor_node) in enumerate(home_grammar.rule_tree):
            if parent_idx is None and ancestor_node is None:
                home_grammar.rule_tree[idx][1] = splitting_idx
                home_grammar.rule_tree[idx][2] = home_node
                break
        else:
            raise AssertionError('never find the root rule')

        home_grammar.rule_tree += [[splitting_rule, None, None]]
        home_grammar.rule_list += [splitting_rule]
        home_grammar.rule_dict[splitting_rule.lhs] = splitting_rule
    else:  # splitting rule already exists
        splitting_rule, = home_grammar.rule_dict[S]
        splitting_idx = home_grammar.find_rule(splitting_rule, where='rule_tree')
        home_node = '0'
        away_node = chr(ord(max(splitting_rule.graph.nodes())) + 1)

        splitting_rule.graph.add_node(away_node, b_deg=0, label=min(lhs for lhs in away_grammar.rule_dict))

    suture_grammars()

    # offset = len(home_grammar.rule_tree)
    # merge_rules(home_grammar, away_grammar, parallel=parallel)

    # # shift the indices of the away decomposition
    # for idx, _ in enumerate(away_grammar.rule_tree):
    #     if away_grammar.rule_tree[idx][1] is not None:
    #         away_grammar.rule_tree[idx][1] += offset
    #     else:
    #         away_grammar.rule_tree[idx][1] = splitting_idx
    #         away_grammar.rule_tree[idx][2] = away_node

    # # shift the indices of the away covering_idx map
    # for node in away_grammar.covering_idx:
    #     away_grammar.covering_idx[node] += offset

    # # append the sub-decomposition to the super-decomposition
    # home_grammar.rule_tree += away_grammar.rule_tree

    # assert len(home_grammar.covering_idx.keys() & away_grammar.covering_idx.keys()) == 0
    # home_grammar.covering_idx |= away_grammar.covering_idx

    return home_grammar


def incise_grammars(home_grammar: VRG, away_grammar: VRG, frontier: set[tuple[int, int]]) -> tuple[VRG, VRG, int, str]:
    # frontier = {(u if u in home_grammar.rule_source else v, v if u in home_grammar.rule_source else u)
    #             for u, v in frontier}

    # u is old => covered by the home_grammar
    # v is new => covered by the away_grammar
    if len(frontier) == 0:
        assert min(lhs for lhs in away_grammar.rule_dict) == 0
        assert min(lhs for lhs in home_grammar.rule_dict) <= 0

        time = max(rule.time for rule in home_grammar.rule_list + away_grammar.rule_list)
        S = min(lhs for lhs in home_grammar.rule_dict)

        if S == 0:  # create & incorporate a splitting rule
            splitting_rule = create_splitting_rule([home_grammar, away_grammar], time)
            splitting_idx = len(home_grammar.rule_tree)
            home_node = '0'
            away_node = '1'

            # make the root of the old decomposition point to the new root
            for idx, (_, parent_idx, ancestor_node) in enumerate(home_grammar.rule_tree):
                if parent_idx is None and ancestor_node is None:
                    home_grammar.rule_tree[idx][1] = splitting_idx
                    home_grammar.rule_tree[idx][2] = home_node
                    break
            else:
                raise AssertionError('never find the root rule')

            home_grammar.rule_tree += [[splitting_rule, None, None]]
            home_grammar.rule_list += [splitting_rule]
            home_grammar.rule_dict[splitting_rule.lhs] = splitting_rule
        else:  # splitting rule already exists
            splitting_rule, = home_grammar.rule_dict[S]
            cover_idx = home_grammar.find_rule(splitting_rule, where='rule_tree')
            home_node = '0'
            away_node = chr(ord(max(splitting_rule.graph.nodes())) + 1)

            splitting_rule.graph.add_node(away_node, b_deg=0, label=min(lhs for lhs in away_grammar.rule_dict))
    else:
        # add additional boundary degrees to the proper nodes in the root rule of the away decomposition
        for _, v in frontier:
            covering_rule_v, cover_idx, ancestor_v = ancestor(v, away_grammar)

            if away_grammar.rule_tree[cover_idx][1] is None and away_grammar.rule_tree[cover_idx][2] is None:
                pass
            else:  # if covering_rule_v is not the root, search for it
                while True:
                    next_idx = away_grammar.rule_tree[cover_idx][1]
                    if away_grammar.rule_tree[next_idx][1] is not None and away_grammar.rule_tree[next_idx][2] is not None:
                        covering_rule_v = away_grammar.rule_tree[next_idx][0]
                        cover_idx = next_idx
                        ancestor_v = away_grammar.rule_tree[next_idx][2]
                    else:
                        covering_rule_v = away_grammar.rule_tree[next_idx][0]
                        break

            covering_rule_v.graph.nodes[ancestor_v]['b_deg'] += 1

            lhs, idx = away_grammar.find_rule(covering_rule_v, where='rule_dict')
            del away_grammar.rule_dict[lhs][idx]

            covering_rule_v.lhs += 1
            if covering_rule_v.lhs in away_grammar.rule_dict:
                away_grammar.rule_dict[covering_rule_v.lhs] += [covering_rule_v]
            else:
                away_grammar.rule_dict[covering_rule_v.lhs] = [covering_rule_v]

        # find the root rule for the new branch (where away_grammar will be grafted in under)
        covering_rule, cover_idx, ancestor_children = common_ancestor({u for u, _ in frontier}, home_grammar)

        # add the nonterminal symbol and connect it to the rest of this root
        away_node = chr(ord(max(covering_rule.graph.nodes())) + 1)
        covering_rule.graph.add_node(away_node, b_deg=0, label=len(frontier))
        for u, _ in frontier:
            ancestor_u = ancestor_children[u]
            covering_rule.graph.add_edge(ancestor_u, away_node)

    return home_grammar, away_grammar, cover_idx, away_node


# when the frontier is nonempty
def suture_grammars(home_grammar: VRG, away_grammar: VRG, cover_idx, away_node, parallel: bool = True) -> VRG:
    offset = len(home_grammar.rule_tree)
    merge_rules(home_grammar, away_grammar, parallel=parallel)

    # shift the indices of the away decomposition
    for idx, (_, parent_idx, ancestor_node) in enumerate(away_grammar.rule_tree):
        if parent_idx is not None and ancestor_node is not None:
            away_grammar.rule_tree[idx][1] += offset
        else:
            away_grammar.rule_tree[idx][1] = cover_idx
            away_grammar.rule_tree[idx][2] = away_node

    # shift the indices of the away covering_idx map
    for node in away_grammar.covering_idx:
        away_grammar.covering_idx[node] += offset

    # append the sub-decomposition to the super-decomposition
    home_grammar.rule_tree += away_grammar.rule_tree

    try:
        assert len(home_grammar.covering_idx.keys() & away_grammar.covering_idx.keys()) == 0
    except:
        print(home_grammar.covering_idx.keys() & away_grammar.covering_idx.keys())
        raise Exception

    home_grammar.covering_idx |= away_grammar.covering_idx

    return home_grammar
