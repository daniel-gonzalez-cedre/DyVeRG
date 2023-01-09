import numpy as np

from cnrg.LightMultiGraph import LightMultiGraph
from cnrg.Rule import Rule
from cnrg.VRG import VRG

from src.utils import boundary_edges


def generate(grammar: VRG, time: int, target_order: int,
             tolerance: float = 0.05, merge_rules: bool = True) -> tuple[LightMultiGraph, list[int]]:

    grammar.compute_rules(time, merge=merge_rules)

    lower_bound = int(target_order * (1 - tolerance))
    upper_bound = int(target_order * (1 + tolerance))
    max_attempts = 1000

    for _ in range(max_attempts):
        g, rule_ordering = generate_graph(grammar.ruledict[time], upper_bound)

        if (g is not None) and (lower_bound <= g.order() <= upper_bound):
            return g, rule_ordering

    raise TimeoutError(f'Generation failed after exceeding {max_attempts} attempts.')


def generate_graph(rules: dict[int, Rule], upper_bound: int) -> tuple[LightMultiGraph, list[int]]:
    node_counter = 1
    rng = np.random.default_rng()

    S = min(rules)  # find the starting symbol
    nonterminals = [0]  # names of nodes in g corresponding to nonterminal symbols
    rule_ordering = []  # idn's of rules in the order they were applied

    g = LightMultiGraph()
    g.add_node(S, label=S)

    while len(nonterminals) > 0:
        if g.order() > upper_bound:
            return None, None

        # choose a nonterminal symbol at random
        nts: int = rng.choice(nonterminals)
        lhs: int = g.nodes[nts]['label']
        rule_candidates: list[Rule] = rules[lhs]

        # select a new rule to apply
        freqs = np.asarray([candidate.frequency for candidate in rule_candidates])
        weights = freqs / np.sum(freqs)
        rule = rng.choice(rule_candidates, p=weights).copy()  # we will have to modify the boundary degrees
        rhs = rule.graph

        rule_ordering.append(rule.idn)
        broken_edges: list[tuple[int, int]] = boundary_edges(g, {nts})
        assert len(broken_edges) == max(0, lhs)

        g.remove_node(nts)
        nonterminals.remove(nts)

        # add all of the nodes from the rule to the graph
        node_map = {}
        for n, d in rhs.nodes(data=True):
            new_node = node_counter
            node_map[n] = new_node
            attr = {'b_deg': d['b_deg']}

            if 'label' in d:
                attr['label'] = d['label']
                nonterminals.append(new_node)

            if 'colors' in d:
                attr['color'] = rng.choice(d['colors'])

            g.add_node(new_node, **attr)
            node_counter += 1

        # add all of the edges from the rule to the graph
        for u, v, d in rhs.edges(data=True):
            attr = {'weight': d['weight']}
            if 'colors' in d:
                attr['color'] = rng.choice(d['colors'])

            g.add_edge(node_map[u], node_map[v], **attr)

        # rewire the broken edges from g to the new structure from the rule
        while len(broken_edges) > 0:
            eidx = rng.choice(len(broken_edges))
            edge = broken_edges.pop(eidx)
            u, v, *d = edge

            # choose a node on the rule's right-hand side to attach this broken edge to
            n = rng.choice([x for x, d in rhs.nodes(data=True) if d['b_deg'] > 0])
            rhs.nodes[n]['b_deg'] -= 1

            # there should never be self-edges on nonterminal symbols
            if u == nts and v != nts:
                u = node_map[n]
            elif u != nts and v == nts:
                v = node_map[n]
            else:
                raise AssertionError(f'investigate: {nts}, {u}, {v}, {edge}')

            # attach the nonterminal we previously selected to the rule node
            g.add_edge(u, v, **dict(*d))

    return g, rule_ordering
