import sys
import networkx as nx
from tqdm import tqdm

sys.path.append('../')

from cnrg.VRG import VRG

from bookkeeping import decompose
from grammar_transitions import graft_grammars
from rule_transitions import update_rule_domestic, update_rule_diplomatic


def update_grammar(grammar: VRG, home_graph: nx.Graph, away_graph: nx.Graph, mode: str = 'joint', mu: int = None):
    if mu is None:
        mu = grammar.mu

    assert mode in ['joint', 'independent']

    charted_grammar = grammar.copy()

    edge_additions = set(away_graph.edges()) - set(home_graph.edges())
    edge_deletions = set(home_graph.edges()) - set(away_graph.edges())

    edges_domestic = {(u, v) for u, v in edge_additions
                      if u in home_graph.nodes() and v in home_graph.nodes()}
    edges_diplomatic = {(u, v) for u, v in edge_additions
                        if bool(u in home_graph.nodes()) != bool(v in home_graph.nodes())}
    edges_foreign = {(u, v) for u, v in edge_additions
                     if u not in home_graph.nodes() and v not in home_graph.nodes()}

    # orient the edges (u, v) so that u ∈ home and v ∈ away
    edges_diplomatic = {(u if u in home_graph.nodes() else v, v if v in away_graph.nodes() else u)
                        for u, v in edges_diplomatic}

    if mode == 'joint':
        uncharted_region = nx.Graph()
        uncharted_region.add_edges_from(edges_diplomatic | edges_foreign)

        uncharted_components: list[frozenset] = [frozenset(nodes) for nodes in nx.connected_components(uncharted_region)]
        uncharted_territories: dict[frozenset, nx.Graph] = {nodes: uncharted_region.subgraph(nodes) for nodes in uncharted_components}

        conquered_components: list[frozenset] = []

        count = 1
        for nodes, territory in tqdm(uncharted_territories.items(), desc=f'joint changes: {count}', leave=True):
            if territory.order() > mu:
                territory_grammar = decompose(territory, mu=mu)
                frontier = {(u if u in home_graph else v, v if v not in home_graph else u)
                            for (u, v) in edges_diplomatic if (u in territory) or (v in territory)}
                charted_grammar = graft_grammars(charted_grammar, territory_grammar, frontier)

                conquered_components += [nodes]

                conquered_diplomatic = {(u, v) for u, v in edges_diplomatic if u in territory and v in territory}
                conquered_foreign = {(u, v) for u, v in edges_foreign if u in territory and v in territory}

                edges_diplomatic -= conquered_diplomatic
                edges_foreign -= conquered_foreign
                count += 1

        for nodes in conquered_components:
            uncharted_components.remove(nodes)
            del(uncharted_territories[nodes])

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


# deprecated
def update_chunk(grammar: VRG, curr_graph: nx.Graph, next_graph: nx.Graph):
    mu = 4
    updated_grammar = grammar.copy()
    grammar.init_temporal_matrix()
    # seen = set(curr_graph.nodes())

    edge_additions = set(next_graph.edges()) - set(curr_graph.edges())
    # edge_deletions = set(curr_graph.edges()) - set(next_graph.edges())

    # case1 = {(u, v) for u, v in edge_additions
    #          if u in curr_graph.nodes() and v in curr_graph.nodes()}
    case2 = {(u, v) for u, v in edge_additions
             if bool(u in curr_graph.nodes()) != bool(v in curr_graph.nodes())}
    case3 = {(u, v) for u, v in edge_additions
             if u not in curr_graph.nodes() and v not in curr_graph.nodes()}

    unexplored_region = nx.Graph()
    unexplored_region.add_edges_from(case2 | case3)

    new_nodes = [nodes for nodes in nx.connected_components(unexplored_region)]
    unexplored_node_components = [(nodes, unexplored_region.subgraph(nodes)) for nodes in new_nodes]

    for nodes, component in unexplored_node_components:
        if component.order() > mu:  # process the update as a chunk of size at least μ
            component_grammar = decompose(component, mu=mu)
            # frontier = {(u if u not in component else v, v if v in component else u)
            #             for (u, v) in case2 if (u in component) or (v in component)}

            # first coordinate is in the old part of the graph
            # second coordinate is in the new part of the graph
            frontier = {(u if u in curr_graph else v, v if v not in curr_graph else u)
                        for (u, v) in case2 if (u in component) or (v in component)}

            updated_grammar = graft_grammars(updated_grammar, component_grammar, frontier)
            return updated_grammar
            exit()

            pass
        else:  # process the updates one-at-a-time
            pass

    raise NotImplementedError
    return


# deprecated
def update_independent(grammar: VRG, curr_graph: nx.Graph, next_graph: nx.Graph):
    grammar = grammar.copy()

    if not grammar.temporal_matrix():
        grammar.init_temporal_matrix()

    seen = set(curr_graph.nodes())

    for v in seen:
        if v not in grammar.which_rule_source:
            print(v)
            raise AssertionError

    edge_additions = set(next_graph.edges()) - set(curr_graph.edges())
    edge_deletions = set(curr_graph.edges()) - set(next_graph.edges())

    case1 = {(u, v) for u, v in edge_additions
             if u in curr_graph.nodes() and v in curr_graph.nodes()}
    case2 = {(u, v) for u, v in edge_additions
             if bool(u in curr_graph.nodes()) != bool(v in curr_graph.nodes())}
    case3 = {(u, v) for u, v in edge_additions
             if u not in curr_graph.nodes() and v not in curr_graph.nodes()}

    changes = case1 | case2

    count = 0
    while len(changes) > 0:
        for u, v in tqdm(changes, desc=f'additions: {count}', leave=True):
            if u in seen and v in seen:
                # print('case1')
                grammar = update_rule_case1(u, v, grammar, 'add')
            elif u in seen and v not in seen:
                # print('case2.1')
                grammar = update_rule_case2(u, v, grammar, 'add')
            elif u not in seen and v in seen:
                # print('case2.2')
                grammar = update_rule_case2(v, u, grammar, 'add')
            elif u not in seen and v not in seen:
                raise AssertionError
            else:
                raise AssertionError

        for u, v in changes:
            seen |= {u, v}

        changes = {(u, v) for u, v in case3 if u in seen or v in seen}
        case3 -= changes
        count += 1

    for u, v in tqdm(edge_deletions, desc='deletions', leave=True):
        grammar = update_rule_case1(u, v, grammar, 'del')

    return grammar
