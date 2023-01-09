# from joblib import Parallel, delayed

from cnrg.VRG import VRG
from src.bookkeeping import common_ancestor, propagate_ancestors
from src.decomposition import create_splitting_rule


def conjoin_grammars(host_grammar: VRG, parasite_grammar: VRG, frontier: set[tuple[int, int]],
                     t1: int, t2: int, approach: str) -> VRG:
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
            approach = the strategy to use when joining the grammars; either ``anneal`` or ``branch``
                       ``anneal``: we regard both grammars equally
                                   boundary edges from the frontier are added to rules in both grammars; changes propagate up
                                   a new root rule is created and both decompositions are inserted under that rule
                       ``branch``: we consider the second grammar to be conditioned on the first
                                   boundary edges from the frontier are added to rules in the second grammar; changes propagate up
                                   the common ancestor of those frontier edges in the first grammar is found
                                   a nonterminal symbol is added to that rule in the first grammar, whose size is len(frontier)
                                   the second grammar decomposition is inserted as a branch under this rule in the first grammar
    """
    strategy = 'anneal' if len(frontier) == 0 else approach.strip().lower()

    if strategy == 'anneal':
        anneal(host_grammar, parasite_grammar, frontier, t1, t2)
    elif strategy == 'branch':
        branch(host_grammar, parasite_grammar, frontier, t1, t2)
    else:
        raise AssertionError(f'{strategy} is not a valid joining strategy; select one of "branch" or "anneal"')


def prepare(u: int, grammar: VRG, t1: int, t2: int, stop_at: int = -1):
    rule_idx = grammar.cover[t2][u]

    if stop_at == rule_idx:
        return

    metarule, pidx, anode = grammar[rule_idx]
    metarule[t2].lhs = max(1, metarule[t2].lhs + 1)
    metarule[t2].graph.nodes[metarule[t2].alias[u]]['b_deg'] += 1

    assert 'label' not in metarule[t2].graph[metarule[t2].alias[u]]
    propagate_ancestors(anode, pidx, grammar, t1, t2, mode='add', stop_at=stop_at)


def anneal(host_grammar: VRG, parasite_grammar: VRG,
           frontier: set[tuple[int, int]], t1: int, t2: int):
    for u, v in frontier:
        prepare(u, host_grammar, t1, t2)
        prepare(v, parasite_grammar, t1, t2)

    splitting_rule = create_splitting_rule((host_grammar, parasite_grammar), t2)
    splitting_rule[t2].graph.add_edges_from([('0', '1') for _, _ in frontier])

    splitting_rule.idn = len(host_grammar.decomposition)
    host_root_idx = host_grammar.root_idx
    host_grammar[host_root_idx][1] = splitting_rule.idn
    host_grammar[host_root_idx][2] = '0'

    host_grammar.decomposition.append([splitting_rule, None, None])
    offset = len(host_grammar.decomposition)

    for idx, (metarule, pidx, anode) in enumerate(parasite_grammar.decomposition):
        if pidx is None and anode is None:
            parasite_grammar[idx][1] = splitting_rule.idn
            parasite_grammar[idx][2] = '1'
        else:
            parasite_grammar[idx][1] += offset
        metarule.idn += offset

    host_grammar.decomposition += parasite_grammar.decomposition


def branch(host_grammar: VRG, parasite_grammar: VRG,
           frontier: set[tuple[int, int]], t1: int, t2: int):
    for _, v in frontier:
        prepare(v, parasite_grammar, t1, t2)

    branch_idx, branch_metarule, mapping = common_ancestor({u for u, _ in frontier}, host_grammar, t2)
    # branch_metarule.ensure(t1, t2)
    # branch_rule = branch_metarule[t2]

    nts = chr(max(ord(v) for v in branch_metarule[t2].graph.nodes()) + 1)
    branch_metarule[t2].graph.add_node(nts, b_deg=0, label=parasite_grammar.root_rule[t2].lhs)
    # branch_metarule[t2].branch = True  # TODO: debugging

    for u, _ in frontier:
        prepare(u, host_grammar, t1, t2, stop_at=branch_idx)
        branch_metarule[t2].graph.add_edge(mapping[u], nts)

        if 'label' in branch_metarule[t2].graph.nodes[mapping[u]]:
            branch_metarule[t2].graph.nodes[mapping[u]]['label'] += 1

    offset = len(host_grammar.decomposition)

    for idx, (metarule, pidx, anode) in enumerate(parasite_grammar.decomposition):
        if pidx is None and anode is None:
            parasite_grammar[idx][1] = branch_idx
            parasite_grammar[idx][2] = nts
        else:
            parasite_grammar[idx][1] += offset
        metarule.idn += offset

    host_grammar.decomposition += parasite_grammar.decomposition
