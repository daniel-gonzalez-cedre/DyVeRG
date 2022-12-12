# from joblib import Parallel, delayed

from cnrg.VRG import VRG
from src.decomposition import create_splitting_rule, common_ancestor, propagate_ancestors


def conjoin_grammars(host_grammar: VRG, parasite_grammar: VRG, frontier: set[tuple[int, int]], time: int, approach: str) -> VRG:
    """
        Joins two grammars in the manner described by the ``approach``.
        Both grammar objects are modified (not copied) in-place; the final joined grammar resides is referred to by the first argument.

        Required arguments:
            host_grammar = the grammar treated as a prior
            parasite_grammar = the grammar treated as an update
            frontier = a set of edges forming a cut on the input graph such that
                       the two halves of the graph induced partition the nodes
                       based on which of the two grammars they are covered by
            time = timestamp corresponding to these changes in the dataset
            approach = the strategy to use when joining the grammars; either ``branch`` or ``anneal``
                       ``branch``: we consider the second grammar to be conditioned on the first
                                   boundary edges from the frontier are added to rules in the second grammar; changes propagate up
                                   the common ancestor of those frontier edges in the first grammar is found
                                   a nonterminal symbol is added to that rule in the first grammar, whose size is len(frontier)
                                   the second grammar decomposition is inserted as a branch under this rule in the first grammar
                       ``anneal``: we regard both grammars equally
                                   boundary edges from the frontier are added to rules in both grammars; changes propagate up
                                   a new root rule is created and both decompositions are inserted under that rule
    """
    if len(frontier) == 0:
        strategy = 'anneal'
    else:
        strategy = approach.strip().lower()

    if strategy == 'anneal':
        anneal(host_grammar, parasite_grammar, frontier, time)
    elif strategy == 'branch':
        branch(host_grammar, parasite_grammar, frontier, time)
    else:
        raise AssertionError(f'{strategy} is not a valid joining strategy; select one of "branch" or "anneal"')


def prepare(u: int, grammar: VRG, time: int, edit: bool, stop_at: int = -1):
    rule_idx = grammar.cover[u]

    if stop_at == rule_idx:
        return

    rule, pidx, anode = grammar[rule_idx]
    rule_u = rule.alias[u]

    rule.lhs = rule.lhs + 1 if rule.lhs >= 0 else 1
    rule.time_changed = time
    rule.graph.nodes[rule_u]['b_deg'] += 1

    if edit:
        rule.edit_dist += 1

    assert 'label' not in rule.graph[rule_u]
    propagate_ancestors(anode, pidx, grammar, time, edit=edit, stop_at=stop_at)


def anneal(host_grammar: VRG, parasite_grammar: VRG, frontier: set[tuple[int, int]], time: int):
    for u, v in frontier:
        prepare(u, host_grammar, time, edit=True)
        prepare(v, parasite_grammar, time, edit=False)

    splitting_rule = create_splitting_rule((host_grammar, parasite_grammar), time)
    splitting_rule.idn = len(host_grammar.decomposition)

    splitting_rule.graph.add_edges_from([('0', '1') for _, _ in frontier])

    for idx, (_, pidx, anode) in enumerate(host_grammar.decomposition):
        if pidx is None and anode is None:
            host_grammar[idx][1] = splitting_rule.idn
            host_grammar[idx][2] = '0'
            break

    host_grammar.decomposition.append([splitting_rule, None, None])
    offset = len(host_grammar.decomposition)

    for idx, (rule, pidx, anode) in enumerate(parasite_grammar.decomposition):
        if pidx is None and anode is None:
            parasite_grammar[idx][1] = splitting_rule.idn
            parasite_grammar[idx][2] = '1'
        else:
            parasite_grammar[idx][1] += offset
        rule.idn += offset

    host_grammar.decomposition += parasite_grammar.decomposition


def branch(host_grammar: VRG, parasite_grammar: VRG, frontier: set[tuple[int, int]], time: int):
    for _, v in frontier:
        prepare(v, parasite_grammar, time, edit=False)

    branch_idx, branch_rule, mapping = common_ancestor({u for u, _ in frontier}, host_grammar)
    branch_rule.time_changed = time

    nts = chr(max(ord(v) for v in branch_rule.graph.nodes()) + 1)

    branch_rule.graph.add_node(nts, b_deg=0, label=parasite_grammar.root_rule.lhs)
    branch_rule.time_changed = time
    branch_rule.branch = True  # TODO: debugging

    for u, _ in frontier:
        prepare(u, host_grammar, time, edit=True, stop_at=branch_idx)
        branch_rule.graph.add_edge(mapping[u], nts)

        if 'label' in branch_rule.graph.nodes[mapping[u]]:
            branch_rule.graph.nodes[mapping[u]]['label'] += 1

    offset = len(host_grammar.decomposition)

    for idx, (rule, pidx, anode) in enumerate(parasite_grammar.decomposition):
        if pidx is None and anode is None:
            parasite_grammar[idx][1] = branch_idx
            parasite_grammar[idx][2] = nts
        else:
            parasite_grammar[idx][1] += offset
        rule.idn += offset

    host_grammar.decomposition += parasite_grammar.decomposition
