import sys

sys.path.append('..')

from joblib import Parallel, delayed

from cnrg.VRG import VRG
from decomposition import create_splitting_rule, assimilate_rules, ancestor, common_ancestor, propagate_ancestors, propagate_descendants


def conjoin_grammars(host_grammar: VRG, parasite_grammar: VRG, frontier: set[tuple[int, int]]) -> VRG:
    """
        Joins two grammars in such a way that one grammar is ``conditioned`` on the other.

        Required arguments:
            host_grammar: VRG = the grammar treated as a prior
            parasite_grammar: VRG = the grammar treated as an update
            frontier: {(int, int) for ...} = a set of edges forming a cut on the input graph such that
                                             the two halves of the graph induced partition the nodes
                                             based on which of the two grammars they are covered by
            root: int = the LHS of the root of the decomposition

        Returns:
            The joined grammar.
    """
    host_grammar, parasite_grammar, attachment_idx, parasite_node = incise_grammars(host_grammar, parasite_grammar, frontier)
    conjoined_grammar = suture_grammars(host_grammar, parasite_grammar, attachment_idx, parasite_node)

    return conjoined_grammar


def incise_grammars(host_grammar: VRG, parasite_grammar: VRG, frontier: set[tuple[int, int]]) -> tuple[VRG, VRG, int, str]:
    # u is old => covered by the host_grammar
    # v is new => covered by the parasite_grammar
    if len(frontier) == 0:
        assert parasite_grammar.root_rule.lhs == 0
        assert host_grammar.root_rule.lhs <= 0

        time = max(rule.time for rule in host_grammar.rule_list + parasite_grammar.rule_list)
        S = host_grammar.root_rule.lhs  # TODO: check this

        if S == 0:  # create & incorporate a splitting rule
            splitting_rule = create_splitting_rule([host_grammar, parasite_grammar], time)
            attachment_idx = len(host_grammar.rule_tree)
            host_node = '0'
            parasite_node = '1'

            # make the root of the old decomposition point to the new root
            host_root_idx = host_grammar.root_idx
            host_grammar.rule_tree[host_root_idx][1] = attachment_idx
            host_grammar.rule_tree[host_root_idx][2] = host_node

            # add the splitting rule to the host grammar
            host_grammar.rule_tree += [[splitting_rule, None, None]]
            host_grammar.rule_list += [splitting_rule]
            host_grammar.rule_dict[splitting_rule.lhs] = splitting_rule
        else:  # splitting rule already exists
            attachment_idx, splitting_rule = host_grammar.root
            parasite_node = chr(ord(max(splitting_rule.graph.nodes())) + 1)

            # splitting_rule.graph.add_node(parasite_node, b_deg=0, label=min(lhs for lhs in parasite_grammar.rule_dict))
            splitting_rule.graph.add_node(parasite_node, b_deg=0, label=parasite_grammar.root_rule.lhs)
    else:
        # add additional boundary degrees to the proper nodes in the root rule of the away decomposition
        for _, v in frontier:
            _, idx_v, ancestor_v = ancestor(v, parasite_grammar)
            propagate_ancestors(ancestor_v, idx_v, parasite_grammar)

        assert len(frontier) == parasite_grammar.root_rule.lhs

        # find the root rule for the new branch (where parasite_grammar will be grafted in under)
        attachment_rule, attachment_idx, ancestor_children = common_ancestor({u for u, _ in frontier}, host_grammar)

        # add the nonterminal symbol and connect it to the rest of this root
        parasite_node = chr(ord(max(attachment_rule.graph.nodes())) + 1)
        attachment_rule.graph.add_node(parasite_node, b_deg=0, label=parasite_grammar.root_rule.lhs)

        for u, _ in frontier:
            ancestor_u = ancestor_children[u]
            attachment_rule.graph.add_edge(ancestor_u, parasite_node)

            for node in (ancestor_u, parasite_node):
                if 'label' in attachment_rule.graph.nodes[node]:
                    propagate_descendants(node, attachment_idx, host_grammar)

    return host_grammar, parasite_grammar, attachment_idx, parasite_node


# when the frontier is nonempty
def suture_grammars(host_grammar: VRG, parasite_grammar: VRG, attachment_idx: int, parasite_node: str, parallel: bool = True) -> VRG:
    offset = len(host_grammar.rule_tree)
    assimilate_rules(host_grammar, parasite_grammar, parallel=parallel)

    # shift the indices of the parasite decomposition
    for idx, (_, parent_idx, ancestor_node) in enumerate(parasite_grammar.rule_tree):
        if parent_idx is not None and ancestor_node is not None:
            parasite_grammar.rule_tree[idx][1] += offset

    # point the old root to the attachment site
    parasite_root_idx = parasite_grammar.root_idx
    parasite_grammar.rule_tree[parasite_root_idx][1] = attachment_idx
    parasite_grammar.rule_tree[parasite_root_idx][2] = parasite_node

    # shift the indices of the parasite covering_idx map
    for node in parasite_grammar.covering_idx:
        parasite_grammar.covering_idx[node] += offset

    # append the sub-decomposition to the super-decomposition
    # host_grammar.rule_tree += parasite_grammar.rule_tree
    for node in parasite_grammar.rule_tree:
        host_grammar.rule_tree.append(node)

    for _, pidx, _ in host_grammar.rule_tree:
        if pidx:
            assert pidx <= len(host_grammar.rule_tree)

    assert len(host_grammar.covering_idx.keys() & parasite_grammar.covering_idx.keys()) == 0

    host_grammar.covering_idx |= parasite_grammar.covering_idx

    return host_grammar
