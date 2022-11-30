import pdb
import logging
import random
from typing import List, Dict, Tuple, Any

import numpy as np

from cnrg.LightMultiGraph import LightMultiGraph
from cnrg.Rule import PartRule
from cnrg.globals import find_boundary_edges


def generate_graph(target_n: int, rule_dict: Dict, tolerance_bounds: float = 0.05) -> Tuple[LightMultiGraph, List[int]]:
    """
        Generates graphs
        :param target_n: number of nodes to target
        :param tolerance_bounds: bounds of tolerance - accept graphs with order ∈ [target_n * (1 - tolerance), target_n * (1 + tolerance)]
        :param rule_dict: dictionary of rules
        :return:
    """
    lower_bound = int(target_n * (1 - tolerance_bounds))
    upper_bound = int(target_n * (1 + tolerance_bounds))
    max_trials = 1_000
    num_trials = 0
    while True:
        if num_trials > max_trials:
            raise TimeoutError(f'Generation failed in {max_trials} steps')

        g, rule_ordering = _generate_graph(rule_dict=rule_dict, upper_bound=upper_bound)
        if g is None:  # early termination
            continue
        if lower_bound <= g.order() <= upper_bound:  # if the number of nodes falls in bounds,
            break

        num_trials += 1

    if num_trials > 1:
        print(f'Graph generated in {num_trials} tries.')
    print(f'Generated graph: n = {g.order()}, m = {g.size()}')

    return g, rule_ordering


def _generate_graph(rule_dict: Dict[int, List[PartRule]], upper_bound: int) -> Any:
    """
    Create a new graph from the VRG at random
    Returns None if the nodes in generated graph exceeds upper_bound
    :return: newly generated graph
    """
    node_counter = 1

    # find the starting rule
    S = min(rule_dict)
    new_g = LightMultiGraph()
    new_g.add_node(S, label=S)
    # new_g.add_node(0, label=0)

    # non_terminals = {0}
    non_terminals = {S}  # TODO: why is this a set?
    rule_ordering = []  # list of rule.idn in the order they were fired

    while len(non_terminals) > 0:
        if new_g.order() > upper_bound:  # graph got too large; abort
            return None, None

        # choose a nonterminal symbol at random
        nts = random.sample(non_terminals, 1)[0]
        lhs = new_g.nodes[nts]['label']

        # select a new rule to apply at the chosen nonterminal symbol
        rule_candidates = rule_dict[lhs]
        if len(rule_candidates) == 1:
            rule = rule_candidates[0]
        else:
            weights = np.array([candidate.frequency for candidate in rule_candidates])
            weights = weights / np.sum(weights)  # normalize into probabilities
            idx = int(np.random.choice(range(len(rule_candidates)), size=1, p=weights))  # pick based on probability
            rule = rule_candidates[idx]

        rhs = rule.graph

        # logging.debug(f'firing rule {rule.idn}, selecting node {nts} with label: {lhs}')
        rule_ordering.append(rule.idn)
        broken_edges = find_boundary_edges(new_g, {nts})
        assert len(broken_edges) == lhs if lhs >= 0 else True

        # get ready to replace the chosen nonterminal with the RHS
        new_g.remove_node(nts)
        non_terminals.remove(nts)

        # add the nodes from the RHS to the generated graph
        node_map = {}
        for n, d in rhs.nodes(data=True):  # all the nodes are internal
            new_node = node_counter
            node_map[n] = new_node
            attr_dict = {'b_deg': d['b_deg']}

            # if it's a new nonterminal, add it to the set of nonterminal symbols
            if 'label' in d:
                non_terminals.add(new_node)
                attr_dict['label'] = d['label']

            # sample a color for this node if there are colors available
            if 'node_colors' in d.keys():
                attr_dict['color'] = random.sample(d['node_colors'], 1)[0]

            new_g.add_node(new_node, **attr_dict)
            node_counter += 1

        # randomly assign broken edges to boundary edges
        random.shuffle(broken_edges)

        # randomly joining the new boundary edges from the RHS to the rest of the graph - uniformly at random
        for n, d in rhs.nodes(data=True):
            num_boundary_edges = d['b_deg']
            if num_boundary_edges == 0:  # there are no boundary edges incident to that node
                continue

            # assert len(broken_edges) >= num_boundary_edges
            # debugging
            if False:
                try:
                    assert len(broken_edges) >= num_boundary_edges
                except AssertionError as E:
                    # pdb.set_trace()
                    raise AssertionError from E

            edge_candidates = broken_edges[:num_boundary_edges]  # picking the first batch of broken edges
            broken_edges = broken_edges[num_boundary_edges:]  # removing them from future consideration

            for e in edge_candidates:  # each edge is either (nts, v(, colors)) or (u, nts(, colors))
                if len(e) == 2:
                    u, v = e
                else:
                    u, v, d = e

                if u == nts:
                    u = node_map[n]
                else:
                    v = node_map[n]

                # logging.debug(f'adding broken edge ({u}, {v})')
                if len(e) == 2:
                    new_g.add_edge(u, v)
                else:
                    new_g.add_edge(u, v, attr_dict=d)

        # adding the RHS to the new graph
        for u, v, d in rhs.graph.edges(data=True):
            edge_multiplicity = d['weight']
            if 'edge_colors' in d.keys():
                edge_color = random.sample(d['edge_colors'], 1)[0]
                new_g.add_edge(node_map[u], node_map[v], weight=edge_multiplicity, edge_color=edge_color)
            else:
                new_g.add_edge(node_map[u], node_map[v], weight=edge_multiplicity)
            # logging.debug(f'adding RHS internal edge ({node_map[u]}, {node_map[v]}) wt: {edge_multiplicity}')

    return new_g, rule_ordering
