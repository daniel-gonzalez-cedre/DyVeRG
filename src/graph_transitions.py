import sys

sys.path.append('../')

import networkx as nx
from tqdm import tqdm

from cnrg.VRG import VRG
from utils import silence
from bookkeeping import decompose
from grammar_transitions import graft_grammars
from rule_transitions import update_rule_domestic, update_rule_diplomatic


def update_grammar(grammar: VRG, home_graph: nx.Graph, away_graph: nx.Graph,
                   time: int, mode: str = 'joint', mu: int = None):
    if mu is None:
        mu = grammar.mu

    assert mode in ['joint', 'j', 'independent', 'indep', 'i']

    charted_grammar = grammar.copy()

    edge_additions = set(away_graph.edges()) - set(home_graph.edges())
    # edge_deletions = set(home_graph.edges()) - set(away_graph.edges())

    edges_domestic = {(u, v) for u, v in edge_additions
                      if u in home_graph.nodes() and v in home_graph.nodes()}
    edges_diplomatic = {(u, v) for u, v in edge_additions
                        if bool(u in home_graph.nodes()) != bool(v in home_graph.nodes())}
    edges_foreign = {(u, v) for u, v in edge_additions
                     if u not in home_graph.nodes() and v not in home_graph.nodes()}

    # orient the edges (u, v) so that u ∈ home and v ∈ away
    edges_diplomatic = {(u if u in home_graph.nodes() else v, v if v not in home_graph.nodes() else u)
                        for u, v in edges_diplomatic}

    if mode in ['joint', 'j']:
        uncharted_region = nx.Graph()
        uncharted_region.add_edges_from(edges_diplomatic | edges_foreign)

        uncharted_components: list[frozenset] = [frozenset(nodes) for nodes in nx.connected_components(uncharted_region)]
        uncharted_territories: dict[frozenset, nx.Graph] = {nodes: uncharted_region.subgraph(nodes) for nodes in uncharted_components}

        conquered_components: list[frozenset] = []

        count = 1
        for nodes, territory in tqdm(uncharted_territories.items(), desc=f'joint changes: {count}', leave=True):
            if territory.order() > 0:

                with silence():
                    territory_grammar = decompose(territory, time=time, mu=mu)

                frontier = {(u if u in home_graph else v, v if v not in home_graph else u)
                            for (u, v) in edges_diplomatic
                            if (u in territory) or (v in territory)}

                # problems = [u for u in territory.nodes() if u not in charted_grammar.rule_source]
                # print(len(problems), territory.order())
                # exit()

                for u, v in frontier:
                    assert u in charted_grammar.rule_source and v in territory_grammar.rule_source

                charted_grammar = graft_grammars(charted_grammar, territory_grammar, frontier)

                conquered_components += [nodes]

                conquered_diplomatic = {(u, v) for u, v in edges_diplomatic if u in territory and v in territory}
                conquered_foreign = {(u, v) for u, v in edges_foreign if u in territory and v in territory}

                edges_diplomatic -= conquered_diplomatic
                edges_foreign -= conquered_foreign
                count += 1

        for nodes in conquered_components:
            uncharted_components.remove(nodes)
            del uncharted_territories[nodes]

        for nodes, territory in uncharted_territories.items():
            for u, v in territory.edges():
                if u in home_graph.nodes() and v in home_graph.nodes():
                    edges_domestic |= {(u, v)}
                elif bool(u in home_graph.nodes()) != bool(v in home_graph.nodes()):
                    edges_diplomatic |= {(u, v)}
                elif u not in home_graph.nodes() and v not in home_graph.nodes():
                    edges_foreign |= {(u, v)}
                else:
                    raise AssertionError(f'{u}, {v}')
    # !!!!!!!
    # for key, val in charted_grammar.rule_source.items():
    #     if val >= len(charted_grammar.rule_tree):
    #         print(key, val)

    # for idx, (rule, parent_idx, which_idx) in enumerate(charted_grammar.rule_tree):
    #     if parent_idx is not None and parent_idx >= len(charted_grammar.rule_tree):
    #         print('!!!!!::', idx, parent_idx)

    # print(len(charted_grammar.rule_tree))
    # !!!!!!!

    charted_grammar.init_temporal_matrix()

    conquered = set(home_graph.nodes())
    changes = edges_domestic | edges_diplomatic

    count = 1
    while len(changes) > 0:
        for u, v in tqdm(changes, desc=f'additions: {count}', leave=True):
            if u in conquered and v in conquered:
                charted_grammar = update_rule_domestic(charted_grammar, u, v, 'add')
            elif u in conquered and v not in conquered:
                charted_grammar = update_rule_diplomatic(charted_grammar, u, v, 'add')
            elif u not in conquered and v in conquered:
                charted_grammar = update_rule_diplomatic(charted_grammar, v, u, 'add')
            else:
                raise AssertionError(f'{u}, {v}')

        for u, v in changes:
            conquered |= {u, v}

        changes = {(u, v) for u, v in edges_foreign if u in conquered or v in conquered}
        edges_foreign -= changes
        count += 1

    # for u, v in tqdm(edge_deletions, desc='deletions', leave=True):
    #     charted_grammar = update_rule_domestic(charted_grammar, u, v, 'del')

    charted_grammar.calculate_cost()

    return charted_grammar
